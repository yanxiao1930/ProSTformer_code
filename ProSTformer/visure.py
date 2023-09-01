import models
import torch
import logging
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader,SequentialSampler, RandomSampler,TensorDataset,Dataset,SubsetRandomSampler
from einops import rearrange, repeat
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, default='nyctaxi')
    parser.add_argument('--train_batch_size', type=int, default=32) #可以试试512
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--pretrained_model_path', type=str, default='./saved_models/index_nyctaxi_30/pytorch_model.bin')
    parser.add_argument('--epoch', type=int, default=1000) #24小时训练完
    parser.add_argument('--early_stop_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--display_steps', type=int, default=100) #可以试试10
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--kernel', type=int, default=4)
    parser.add_argument('--name', type=str, default=False)
    parser.add_argument('--time', type=str, default='30')
    parser.add_argument('--interval', type=int, default=48)
    parser.add_argument('--frame', type=int, default=4)



    args = parser.parse_args()

    # 设置参数
    if args.name==False:
        args.output_dir = "./saved_models/index_{}_{}".format(args.index,args.time)
    else:
        args.output_dir ="./saved_models/{}".format(args.name)
    # os.system("mkdir -p {}".format(args.output_dir))  # 创建目录
    # if args.pretrained_model_path != False:
    #     args.output_dir=args.output_dir+'_pretrained'
    os.makedirs(args.output_dir,exist_ok=True)
    print(args.output_dir)

    #日志
    logger = logging.getLogger('train')
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    handler = logging.FileHandler(os.path.join(args.output_dir, "train_log.txt"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s -   %(message)s')  # 这里可以更改logging在储存到txt的结构
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("Argument %s", args)

    # 读取数据
    # def shrink_size(data_x, data_y, kernel_size):
    #     b, f, c, _, _ = data_x.shape
    #     data_x = rearrange(data_x, 'b f c h w -> (b f c) h w')
    #     data_y = rearrange(data_y, 'b c h w -> (b c) h w')
    #     pool = nn.AvgPool2d((kernel_size))
    #     data_x = pool(data_x) * kernel_size[0] * kernel_size[1]
    #     data_y = pool(data_y) * kernel_size[0] * kernel_size[1]
    #     data_x = rearrange(data_x, '(b f c) h w -> b f c h w ', b=b, f=f, c=c)
    #     data_y = rearrange(data_y, '(b c) h w -> b c h w ', b=b, c=c)
    #     return data_x, data_y
    #
    #
    # if args.index == 'nycbike' or args.index == 'nycbike_shrink':
    #     external = np.load('../../data_nycbike/ext_features.npy', allow_pickle=True)
    #     external = external[:, np.newaxis, :]
    #     external = torch.tensor(external, dtype=torch.float32)
    #
    # if args.index == 'nyctaxi' or args.index == 'nyctaxi_shrink':
    #     external = np.load('../../data_nyctaxi/ext_features.npy', allow_pickle=True)
    #     external = external[:, np.newaxis, :]
    #     external = torch.tensor(external, dtype=torch.float32)
    #
    # if args.index == 'bjtaxi' or args.index == 'bjtaxi_shrink':
    #     train_video = np.load('../../data_bjtaxi/data_train_x.npy', allow_pickle=True)
    #     train_y = np.load('../../data_bjtaxi/data_train_y.npy', allow_pickle=True)
    #     dev_video = np.load('../../data_bjtaxi/data_test_x.npy', allow_pickle=True)
    #     dev_y = np.load('../../data_bjtaxi/data_test_y.npy', allow_pickle=True)
    #     train_y = torch.tensor(train_y, dtype=torch.float32)
    #     dev_y = torch.tensor(dev_y, dtype=torch.float32)
    #     train_external = np.load('../../data_bjtaxi/train_ext.npy', allow_pickle=True)
    #     dev_external = np.load('../../data_bjtaxi/test_ext.npy', allow_pickle=True)
    #     train_external = train_external[:, np.newaxis, :]
    #     train_external = torch.tensor(train_external, dtype=torch.float32)
    #     dev_external = dev_external[:, np.newaxis, :]
    #     dev_external = torch.tensor(dev_external, dtype=torch.float32)
    #     ss = MinMaxScaler(feature_range=(-1, 1))
    #     # train dataset
    #     b, f, c, h, w = train_video.shape
    #     train_video = rearrange(train_video, 'b f c h w -> b (f c h w)')  # video做min-max标准化
    #     train_video = ss.fit_transform(train_video)
    #     train_video = rearrange(train_video, 'b (f c h w) -> b f c h w', c=c, h=h, w=w)
    #     train_video = torch.tensor(train_video, dtype=torch.float32)
    #     b, f, c, h, w = train_video.shape
    #     # dev dataset
    #     dev_video = rearrange(dev_video, 'b f c h w -> b (f c h w)')  # video做min-max标准化
    #     dev_video = ss.fit_transform(dev_video)
    #     dev_video = rearrange(dev_video, 'b (f c h w) -> b f c h w', c=c, h=h, w=w)
    #     dev_video = torch.tensor(dev_video, dtype=torch.float32)
    #     # shrink
    #     if args.index == 'bjtaxi_shrink':
    #         train_video, train_y = shrink_size(train_video, train_y, (2, 2))  # h:16, w:16
    #         dev_video, dev_y = shrink_size(dev_video, dev_y, (2, 2))  # h:16, w:16
    #     logger.info("grid map size : {}".format(train_video.shape))
    #     # 生成数据集)
    #     print(train_video.shape, train_external.shape, train_y.shape)
    #     train_dataset = TensorDataset(train_video, train_external, train_y)
    #     dev_dataset = [dev_video, dev_external, dev_y]
    #
    # # 生成训练集和验证集
    # if args.index == 'nycbike' or args.index == 'nycbike_shrink' or args.index == 'nyctaxi' or args.index == 'nyctaxi_shrink':
    #     # print(external.shape)
    #     ss = MinMaxScaler(feature_range=(-1, 1))
    #     video = np.load('../../data_{}/data_x.npy'.format(args.index.replace('_shrink', '')), allow_pickle=True)
    #     b, f, c, h, w = video.shape
    #     y = torch.tensor(
    #         np.load('../../data_{}/data_y.npy'.format(args.index.replace('_shrink', '')), allow_pickle=True),
    #         dtype=torch.float32)
    #     video = rearrange(video, 'b f c h w -> b (f c h w)')  # video做min-max标准化
    #     video = ss.fit_transform(video)
    #     video = rearrange(video, 'b (f c h w) -> b f c h w', c=c, h=h, w=w)
    #     video = torch.tensor(video, dtype=torch.float32)
    #     if args.index == 'nyctaxi_shrink' or args.index == 'nycbike_shrink':
    #         video, y = shrink_size(video, y, (4, 4))  # h:12, w:16
    #     logger.info("grid map size : {}".format(video.shape))
    #
    #     min_index = 4 * 7 * 48  # 划分点：最后4周测试集，往前推4周是验证集，再往前是测试集
    #     # print(video.shape,external.shape,y.shape)
    #     train_dataset = TensorDataset(video[0:-1 * min_index], external[0:-1 * min_index], y[0:-1 * min_index])
    #     dev_video = video[-1 * min_index:]
    #     dev_external = external[-1 * min_index:]
    #     dev_y = y[-1 * min_index:]
    #     dev_dataset = [dev_video, dev_external, dev_y]
    # 读取数据
    #shrink size
    def shrink_size(data_x, data_y, kernel_size):
        b, f, c, _, _ = data_x.shape
        data_x = rearrange(data_x, 'b f c h w -> (b f c) h w')
        data_y = rearrange(data_y, 'b c h w -> (b c) h w')
        pool = nn.AvgPool2d((kernel_size))
        data_x = pool(data_x) * kernel_size[0] * kernel_size[1]
        data_y = pool(data_y) * kernel_size[0] * kernel_size[1]
        data_x = rearrange(data_x, '(b f c) h w -> b f c h w ', b=b, f=f, c=c)
        data_y = rearrange(data_y, '(b c) h w -> b c h w ', b=b, c=c)
        return data_x, data_y


    if args.index == 'bjtaxi' or args.index == 'bjtaxi_shrink':
        latest_wea = np.load('../../data_bjtaxi/latest_wea.npy', allow_pickle=True)
        forward = np.load('../../data_bjtaxi/m_data_future.npy', allow_pickle=True)
        video_scale = MinMaxScaler(feature_range=(-1, 1))
        latest_wea[:, [0, -1]] = video_scale.fit_transform(latest_wea[:, [0, -1]])  # 温度和风速min-max标准化
        external = np.concatenate((latest_wea, forward), axis=1)  # 合并wea和day_of_week,holiday
        external = external[:, np.newaxis, :]
        external = torch.tensor(external, dtype=torch.float32)
    if args.index == 'nycbike' or args.index == 'nycbike_shrink' or args.index == 'nycbike_shrink_1':
        external = np.load('../../data_nycbike/ext_features_{}.npy'.format(args.time), allow_pickle=True)
        external = external.astype(np.int32)
        external = external[:, np.newaxis, :]
        file_name='nycbike'

    if args.index == 'nyctaxi' or args.index == 'nyctaxi_shrink' or args.index == 'nyctaxi_shrink_1':
        external = np.load('../../data_nyctaxi/ext_features_{}.npy'.format(args.time), allow_pickle=True)
        external = external[:, np.newaxis, :]
        external=external.astype(np.int32)
        file_name = 'nyctaxi'

    video_scale = MinMaxScaler(feature_range=(-1, 1))
    external_scale = MinMaxScaler(feature_range=(-1, 1))
    X=np.load('../../data_{}/data_x_{}.npy'.format(file_name, args.time), allow_pickle=True)
    Y=torch.tensor(np.load('../../data_{}/data_y_{}.npy'.format(file_name, args.time), allow_pickle=True), dtype=torch.float32)

    min_index = 4 * 7 * args.interval
    val_min_index = 4 * 7 * args.interval  # 划分点：最后4周测试集，往前推1周是验证集，再往前是测试集
    train_min_index = 5 * 7 * args.interval
    b, f, c, h, w = X.shape


    external[min_index : -1 * train_min_index,0,0:2]=external_scale.fit_transform(external[min_index : -1 * train_min_index, 0, 0:2])
    external[-1 * train_min_index : -1 * val_min_index, 0, 0:2] = external_scale.transform(external[-1 * train_min_index : -1 * val_min_index, 0, 0:2])
    external[ -1 * val_min_index:, 0, 0:2] = external_scale.transform(external[ -1 * val_min_index:, 0, 0:2])
    external = torch.tensor(external, dtype=torch.float32)


    X = rearrange(X, 'b f c h w -> b (f c h w)')#video做min-max标准化
    X[0 : -1 * train_min_index]= video_scale.fit_transform(X[0: -1 * train_min_index])
    X[-1 * train_min_index : -1 * val_min_index, ]=video_scale.transform(X[-1 * train_min_index: -1 * val_min_index, ])#验证集norm
    X[-1 * val_min_index:, ] = video_scale.transform(X[-1 * val_min_index:, ])
    X = rearrange(X, 'b (f c h w) -> b f c h w', c=c, h=h, w=w)
    X=torch.tensor(X, dtype=torch.float32)
    print(b, external.shape)


    train_dataset=TensorDataset(X[0:-1 * train_min_index], external[min_index:-1 * train_min_index], Y[0:-1 * train_min_index])
    val_dataset=TensorDataset(X[-1 * train_min_index : -1 * val_min_index], external[-1 * train_min_index : -1 * val_min_index], Y[-1 * train_min_index : -1 * val_min_index])
    test_dataset = TensorDataset(X[-1 * val_min_index:, ], external[-1 * val_min_index:, ], Y[-1 * val_min_index:, ])

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size,
                                  num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=args.train_batch_size,
                                  num_workers=0)
    test_dataloader = DataLoader(test_dataset,  batch_size=args.train_batch_size,shuffle=False,
                                  num_workers=0)



    model = models.ctrNet(args,logger)


    # reload_loss=None

    # 训练模型
    # model.train(train_dataloader, test_dataloader,None)

    # 输出验证集结果
    model.reload()#加载最佳模型参数
    model.model.eval()
    pres, y = model.infer(test_dataloader)
    # for i in range(48*3*7+6*48,48*3*7+7*48):
    os.makedirs('./image',exist_ok=True)
    os.makedirs('./image_lim', exist_ok=True)
    for i in range(0, 48):
        i=i % 48
        plt.imshow(np.rot90(y[i, 1]),cmap='coolwarm')
        vmax=torch.max(y[i, 1])
        vmin = torch.min(y[i, 1])
        cp = plt.colorbar()
        plt.axis('off')
        plt.savefig('./image/outflow_{}.jpg'.format(i), dpi=600, bbox_inches='tight')
        plt.close()

        plt.imshow(np.rot90(y[i, 1]), cmap='coolwarm',vmax=50)
        vmax = torch.max(y[i, 1])
        vmin = torch.min(y[i, 1])
        cp = plt.colorbar()
        plt.axis('off')
        plt.savefig('./image_lim/outflow_{}.jpg'.format(i), dpi=600, bbox_inches='tight')
        plt.close()

        plt.imshow(np.rot90(pres[i, 1]),cmap='coolwarm',vmax=vmax,vmin=vmin)
        cp = plt.colorbar()
        plt.axis('off')
        plt.savefig('./image/pres_outflow_{}.jpg'.format(i), dpi=600, bbox_inches='tight')
        plt.close()

        plt.imshow(np.rot90(pres[i, 1]),cmap='coolwarm',vmax=50)
        cp = plt.colorbar()
        plt.axis('off')
        plt.savefig('./image_lim/pres_outflow_{}.jpg'.format(i), dpi=600, bbox_inches='tight')
        plt.close()








