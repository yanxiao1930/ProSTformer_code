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


class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


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

        
        self.to_q=nn.Linear(dim, inner_dim , bias = True)
        self.to_k = nn.Linear(dim, inner_dim, bias=True)
        self.to_v = nn.Linear(dim, inner_dim, bias=True)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, einops_from, einops_to, **einops_dims):
        h = self.heads
        
        q=self.to_q(x)
        k= self.to_k(x)
        v= self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v)) 

        q *= self.scale
        ( q_), ( k_), ( v_) = map(lambda t: ( t[:, :]), (q, k, v))

        
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        
        out = attn(q_, k_, v_)

        
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        
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
        
        x=torch.cat((external,x),dim=1)
        q=self.to_q(x)
        k= self.to_k(x)
        v= self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v)) 

        q *= self.scale

        (ext_q, q_), (ext_k, k_), (ext_v, v_) = map(lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))
        
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        
        r = q_.shape[0] // ext_k.shape[0]
        ext_k, ext_v,ext_q = map(lambda t: repeat(t, 'b n d -> (b r) n d', r = r), (ext_k, ext_v,ext_q))

        k_ = torch.cat((ext_k, k_), dim = 1)
        v_ = torch.cat((ext_v, v_), dim = 1)
        q_ = torch.cat((ext_q, q_), dim=1)

        
        out = attn(q_, k_, v_)
        ext_out=out[:,0]
        ext_out = rearrange(ext_out, '(b h r)  d -> b r (h d)', h=h,r=r)
        ext_out=ext_out[:,1]
        ext_out=torch.unsqueeze(ext_out,1)
        out = rearrange(out[:,1:], f'{einops_to} -> {einops_from}', **einops_dims)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        return self.to_out(out), self.to_out(ext_out)




class ProSTformer(nn.Module):
    def __init__(
        self,
        dim=128, 
        dim_1=512, 
        exter_dim=28,
        num_frames=4, 
        periods=3, 
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
        

        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1]) 
        num_positions = num_frames * periods * num_patches 
        patch_dim = channels * patch_size[0] *patch_size[1]   
        
        num_patches_1 = (image_size[0] // patch_size_1[0]) *(image_size[1] // patch_size_1[1])   
        self.num_patches_1=num_patches_1
        self.num_patches=num_patches
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.pos_emb = nn.Embedding(num_positions, dim)

        self.exter_embd=nn.Linear(exter_dim, dim_1)
        self.projection=nn.Linear(dim, channels)


        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)), 
                PreNorm(dim_1, Attention(dim_1, dim_head=dim_head_1, heads=heads_1, dropout=attn_dropout)),
                PreNorm(dim, Attention(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)),  
                PreNorm(dim_1, Attention_external(dim_1, dim_head=dim_head_1, heads=heads_1, num_patches=self.num_patches_1, dropout=attn_dropout)),
                PreNorm(dim_1, FeedForward(dim_1, dropout = ff_dropout))
            ]))



    def forward(self, video,external,y=None):
        b, f, _, h, w, device, p1,p2 = *video.shape, video.device, self.patch_size[0],self.patch_size[1]
        video = rearrange(video, 'b pf c (h p1) (w p2) -> b (pf h w) c p1 p2', p1 = self.patch_size_1[0], p2=self.patch_size_1[1]) 
        video = rearrange(video, 'b pfs c (h p1) (w p2) -> b (pfs h w) (c p1 p2)', p1 =self.patch_size[0], p2=self.patch_size[1])
        tokens = self.to_patch_embedding(video)
        x=tokens
        x += self.pos_emb(torch.arange(x.shape[1], device = device))
        external=self.exter_embd(external)
        video=x

        for (spatial_attn,spatial_attn_1,time_attn,time_attn_1, ff) in self.layers:
            #space blocks
            x = spatial_attn(video, 'b (p f s n) d', '(b p f s) n d', s=self.s, n=self.n, p=self.periods, f=self.num_frames) + video
            x=rearrange(x, 'b (p f s n) d -> b (p f s) (n d)', s=self.s, n=self.n, p=self.periods, f=self.num_frames)
            x = spatial_attn_1(x, 'b (p f s) nd', '(b p f) s nd', s=self.s, p=self.periods, f=self.num_frames) + x  
            #time blocks
            x_ = time_attn(video, 'b (p f s n) d', '(b p s n) f d',  s=self.s, n=self.n, p=self.periods, f=self.num_frames) + video
            x_=rearrange(x_, 'b (p f s n) d -> b (p f s) (n d)', s=self.s, n=self.n, p=self.periods, f=self.num_frames)
            x_0,external_0 = time_attn_1(x_, 'b (p f s) nd', '(b s) (p f) nd',external, s=self.s, p=self.periods, f=self.num_frames)    
            x_=x_0+x_
            external=external_0+external
            #fuse
            video=x+x_
            #feedforward
            video = ff(video) + video
            external=ff(external) + external
            video= rearrange(video, 'b (p f s) (n d) -> b (p f s n) d', s=self.s, n=self.n, p=self.periods, f=self.num_frames)  
            
        #take the recent frame
        video = rearrange(video, 'b (p f s n) d-> b (p f) (s n) d', s=self.s, n=self.n, p=self.periods, f=self.num_frames)
        recent=video[:,-1,:,:]
        recent=self.projection( recent)
        #combination
        dec_out = rearrange(recent, 'b (s h w) (c p1 p2) -> b s (h p1) (w p2) c',c=self.channels,p1=self.patch_size[0],p2=self.patch_size[1],h=self.patch_size_1[0]//self.patch_size[0],w=self.patch_size_1[1]//self.patch_size[1])
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
    model = ProSTformer(
        dim =32,
        dim_1=32*12,
        exter_dim=56,
        num_frames=4,  
        periods=3,  
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

    video = torch.randn(1, 12, 2, 12, 16).cuda() 
    exter=torch.randn(1, 1,56).cuda()
    truth = torch.randn(1, 2, 12, 16)
    model(video,exter)
    flops, params = profile(model, inputs=(video,exter))
    flops, params = clever_format([flops, params], '%.3f')
    print('Parameters：',params)
    print('Flops：',flops)


