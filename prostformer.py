import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
import torch.optim as optim
from thop import profile
from thop import clever_format



class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

# feedforward
#
# class GEGLU(nn.Module):
#     def forward(self, x):
#         x, gates = x.chunk(2, dim = -1)
#         return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult ),
            # GEGLU(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

# attention
def attn(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim = -1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = True)
        self.to_q=nn.Linear(dim, inner_dim , bias = True)
        self.to_k = nn.Linear(dim, inner_dim, bias=True)
        self.to_v = nn.Linear(dim, inner_dim, bias=True)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, einops_from, einops_to, **einops_dims):
        h = self.heads
        # q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q=self.to_q(x)
        k= self.to_k(x)
        v= self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v)) #n是序列长度等于（f*n），h是head数量，d是(q,k,v)的分向量维度，这里把head和batch合并到一个维度了

        q *= self.scale

        # splice out clsernal token at index 1
        # (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:4], t[:, 4:]), (q, k, v))
        ( q_), ( k_), ( v_) = map(lambda t: ( t[:, :]), (q, k, v))
        # let classification token attend to key / values of all patches across time and space
        # cls_out = attn(cls_q, k, v)

        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))#把q,k,v重组 ，以time为例：单个做时间att,batch做空间att

        # expand cls token keys and values across time or space and concat
        # r = q_.shape[0] // cls_k.shape[0]
        # cls_k, cls_v = map(lambda t: repeat(t, 'b n d -> (b r) n d', r = r), (cls_k, cls_v))
        #
        # k_ = torch.cat((cls_k, k_), dim = 1)
        # v_ = torch.cat((cls_v, v_), dim = 1)

        # attention
        out = attn(q_, k_, v_)

        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)#做完att后back回之前的形状


        # concat back the cls token
        # out = torch.cat((cls_out, out), dim = 1)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        # combine heads out
        return self.to_out(out)

class Attention_external(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        num_patches=4
    ):
        super().__init__()
        self.num_patches=num_patches
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = True)
        self.to_q=nn.Linear(dim, inner_dim , bias = True)
        self.to_k = nn.Linear(dim, inner_dim, bias=True)
        self.to_v = nn.Linear(dim, inner_dim, bias=True)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, einops_from, einops_to, external, **einops_dims):
        b,_,_=external.shape
        h = self.heads
        # q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        x=torch.cat((external,x),dim=1)
        q=self.to_q(x)
        k= self.to_k(x)
        v= self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v)) #n是序列长度等于（f*n），h是head数量，d是(q,k,v)的分向量维度，这里把head和batch合并到一个维度了

        q *= self.scale

        # splice out external token at index 1
        (ext_q, q_), (ext_k, k_), (ext_v, v_) = map(lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))
        # (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:self.num_patches], t[:, self.num_patches:]), (q_, k_, v_))
        # ( q_), ( k_), ( v_) = map(lambda t: ( t[:, :]), (q, k, v))
        # let classification token attend to key / values of all patches across time and space
        # ext_out = attn(ext_q, k, v) # external out,因为后面要变形，所以cls只能单独做att
        # ext_out = rearrange(ext_out, '(b h) n d -> b n (h d)', h=h)
        # cls_out = attn(cls_q, k, v) # external out,因为后面要变形，所以cls只能单独做att
        # print(ext_out.shape)

        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))#把q,k,v重组 ，以time为例：单个做时间att,batch做空间att

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // ext_k.shape[0]
        ext_k, ext_v,ext_q = map(lambda t: repeat(t, 'b n d -> (b r) n d', r = r), (ext_k, ext_v,ext_q))
        # r = q_.shape[0] // cls_k.shape[0]
        # cls_q,cls_k = map(lambda t: repeat(t, 'b n d -> (b r) n d', r = r), (cls_q,cls_k))

        k_ = torch.cat((ext_k, k_), dim = 1)
        v_ = torch.cat((ext_v, v_), dim = 1)
        q_ = torch.cat((ext_q, q_), dim=1)

        # attention
        out = attn(q_, k_, v_)
        ext_out=out[:,0]
        ext_out = rearrange(ext_out, '(b h r)  d -> b r (h d)', h=h,r=r)
        ext_out=ext_out[:,1]
        ext_out=torch.unsqueeze(ext_out,1)
        # merge back time or space
        # ext_out = attn(ext_q, k_, v) # external out,因为后面要变形，所以cls只能单独做att
        # ext_out = rearrange(ext_out, '(b h) n d -> b n (h d)', h=h)
        # cls_out = attn(cls_q, k_, v_) # external out,因为后面要变形，所以cls只能单独做att
        # cls_out=rearrange(cls_out, '(b r) s d -> b r s d', r=r)
        # cls_out=cls_out[:,0]
        out = rearrange(out[:,1:], f'{einops_to} -> {einops_from}', **einops_dims)#做完att后back回之前的形状

        # concat back the cls token
        # out = torch.cat((cls_out, out), dim = 1)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        # combine heads out
        return self.to_out(out), self.to_out(ext_out)



# main classes
class TimeSformer(nn.Module):
    def __init__(
        self,
        dim=128, #8*8*(3-2)
        dim_1=512, #16*16*(3-2)
        exter_dim=28,
        num_frames=4, #4
        periods=3, #3
        s=4,
        n=4,
        image_size = (32,32),
        patch_size = (8,8),
        patch_size_1=(16,16),
        channels =2,
        depth = 8,
        heads = 4,
        heads_1=8,
        dim_head = 32,
        dim_head_1=64,
        attn_dropout = 0.1,
        ff_dropout = 0.1
    ):
        super().__init__()
        self.channels=channels
        self.patch_size=patch_size
        self.patch_size_1 = patch_size_1
        self.image_size=image_size
        self.depth=depth
        self.heads=heads
        self.dim_head=dim_head
        self.dim_head_1 = dim_head_1
        self.heads = heads
        self.periods=periods
        self.num_frames=num_frames
        self.exter_dim=exter_dim
        self.s=s
        self.n=n
        self.p1=(image_size[0] // patch_size_1[0])
        self.p2=(image_size[1] // patch_size_1[1])
        # assert s==(image_size[0] // patch_size_1[0]) * (image_size[1] // patch_size_1[1]), 'Image dimensions must be divisible by the patch size.'
        # assert n==(patch_size_1[0] // patch_size[0]) * (patch_size_1[0] // patch_size[0]), 'Image dimensions must be divisible by the patch size.'

        #merge之前的参数
        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1]) #16
        num_positions = num_frames * periods * num_patches #4*3*16
        patch_dim = channels * patch_size[0] *patch_size[1]   #2*8^2
        # merge之后的参数
        num_patches_1 = (image_size[0] // patch_size_1[0]) *(image_size[1] // patch_size_1[1])   # 4
        self.num_patches_1=num_patches_1
        self.num_patches=num_patches
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.pos_emb = nn.Embedding(num_positions, dim)
        # self.cls_token = nn.Parameter(torch.randn(num_patches_1, dim_1)) #decoder用的patch token
        # self.cls_emb = nn.Embedding(num_patches_1, dim_1)

        self.exter_embd=nn.Linear(exter_dim, dim_1)
        self.projection=nn.Linear(dim, channels)

        #merge之后的参数


        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)), #time1
                PreNorm(dim_1, Attention(dim_1, dim_head=dim_head_1, heads=heads_1, dropout=attn_dropout)),#time
                PreNorm(dim, Attention(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)),  # time1
                PreNorm(dim_1, Attention_external(dim_1, dim_head=dim_head_1, heads=heads_1, num_patches=self.num_patches_1, dropout=attn_dropout)),#time
                PreNorm(dim_1, FeedForward(dim_1, dropout = ff_dropout))
            ]))

        # self.encoder_layers = nn.ModuleList([]) #decoder
        # for _ in range(depth_decoder):
        #     self.encoder_layers .append(nn.ModuleList([
        #         PreNorm(dim_1, Deocoder_layer(dim_1, dim_head=dim_head_1, heads=heads_1, dropout=attn_dropout)),
        #         PreNorm(dim_1, FeedForward(dim_1, dropout=ff_dropout))
        #     ]))


    def forward(self, video,external,y=None):
        b, f, _, h, w, *_, device, p1,p2 = *video.shape, video.device, self.patch_size[0],self.patch_size[1]
        # assert h % p1 == 0 and w % p2 == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'

        n = (h // p1) * (w // p2)

        #16分
        video = rearrange(video, 'b pf c (h p1) (w p2) -> b (pf h w) c p1 p2', p1 = self.patch_size_1[0], p2=self.patch_size_1[1]) #划分4个区块
        video = rearrange(video, 'b pfs c (h p1) (w p2) -> b (pfs h w) (c p1 p2)', p1 =self.patch_size[0], p2=self.patch_size[1]) #b pfsn,d
        # print('16分',video.shape)
        tokens = self.to_patch_embedding(video)
        # cls_token= repeat(self.cls_token, 'n d -> b n d', b = b)
        # cls_token= cls_token+self.cls_emb(torch.arange(cls_token.shape[1], device = device))
        x=tokens
        x += self.pos_emb(torch.arange(x.shape[1], device = device))
        # print(external.shape)
        external=self.exter_embd(external)
        # print(external.shape)
        video=x

        #Encoder
        for (spatial_attn,spatial_attn_1,time_attn,time_attn_1, ff) in self.layers:
            x = spatial_attn(video, 'b (p f s n) d', '(b p f s) n d', s=self.s, n=self.n, p=self.periods, f=self.num_frames) + video   #part att
            # print(x.shape)
            x=rearrange(x, 'b (p f s n) d -> b (p f s) (n d)', s=self.s, n=self.n, p=self.periods, f=self.num_frames) #merge
            # print(x.shape)
            x = spatial_attn_1(x, 'b (p f s) nd', '(b p f) s nd', s=self.s, p=self.periods, f=self.num_frames) + x  # 第二维是序列长度，先做时间的attention,所以是序列frame的长度
            # print(x.shape)
            # print(x.shape)


            x_ = time_attn(video, 'b (p f s n) d', '(b p s n) f d',  s=self.s, n=self.n, p=self.periods, f=self.num_frames) + video #period内的att
            # print(x.shape) #(b,12*16=192,128)
            x_=rearrange(x_, 'b (p f s n) d -> b (p f s) (n d)', s=self.s, n=self.n, p=self.periods, f=self.num_frames) #merge
            # print(x.shape) #(b, 12*4, 4*128)
            # x=torch.cat((cls_token,x),dim=1) #只在最后一次做cls attn
            x_0,external_0 = time_attn_1(x_, 'b (p f s) nd', '(b s) (p f) nd',external, s=self.s, p=self.periods, f=self.num_frames)    #再做空间的attention,划分的patch size(n)，所以序列是n
            x_=x_0+x_
            external=external_0+external


            video=x+x_

            video = ff(video) + video
            external=ff(external) + external
            # print(x.shape)
            # cls_token=x[:,0:self.num_patches_1]
            # x=x[:,self.num_patches_1:] #只在最后做cls att
            video= rearrange(video, 'b (p f s) (n d) -> b (p f s n) d', s=self.s, n=self.n, p=self.periods, f=self.num_frames)  # 下一层之前先split
            # print(x.shape)


        video = rearrange(video, 'b (p f s n) d-> b (p f) (s n) d', s=self.s, n=self.n, p=self.periods, f=self.num_frames)

        cls_token=video[:,-1,:,:]
        cls_token=self.projection(cls_token)
        # print(1,cls_token.shape)
        # dec_out = rearrange(cls_token, 'b (h1 w1 h2 w2) d -> b d (h1 h2) (w1 w2)',h1=int(h/self.patch_size_1[0]),w1=int(w/self.patch_size_1[1]),h2=self.patch_size_1[0],w2=self.patch_size_1[1])
        dec_out = rearrange(cls_token, 'b (s h w) (c p1 p2) -> b s (h p1) (w p2) c',c=self.channels,p1=self.patch_size[0],p2=self.patch_size[1],h=self.patch_size_1[0]//self.patch_size[0],w=self.patch_size_1[1]//self.patch_size[1])
        dec_out = rearrange(dec_out, 'b (h w) p1 p2 c -> b c (h p1) (w p2)',p1=self.patch_size_1[0],p2=self.patch_size_1[1],h=self.image_size[0]//self.patch_size_1[0],w=self.image_size[1]//self.patch_size_1[1])


        def MAE(pred, gt):
            mae = torch.abs(pred - gt).mean()
            return mae

        if y is not None:
            loss=F.mse_loss(dec_out, y)
            mae=MAE(dec_out,y)
            return loss,mae
        else:
            pre=dec_out
            return pre



if __name__ == "__main__":
    model = TimeSformer(
        dim =32,
        dim_1=32*12,
        exter_dim=56,
        num_frames=4,  # 4
        periods=3,  # 3
        s=16,
        n=12,
        image_size=(12,16),
        patch_size=(1,1),
        patch_size_1=(3,4),
        channels=2,
        depth=6,
        heads=8,
        heads_1=8,
        dim_head=4,
        dim_head_1=48,
        attn_dropout=0.2,
        ff_dropout=0.2,
    ).cuda()

    video = torch.randn(1, 12, 2, 12, 16).cuda() # (batch x frames x channels x height x width)
    exter=torch.randn(1, 1,56).cuda()
    truth = torch.randn(1, 2, 12, 16)
    model(video,exter)
    flops, params = profile(model, inputs=(video,exter))
    flops, params = clever_format([flops, params], '%.3f')
    print('Parameters：',params)
    print('Flops：',flops)


