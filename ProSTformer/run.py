import models
import torch
import logging
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader,SequentialSampler, RandomSampler,TensorDataset
from einops import rearrange, repeat
import torch.nn.functional as F
from torch import nn, einsum
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, default='nyctaxi')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--pretrained_model_path', type=str, default=False)
    # parser.add_argument('--pretrained_model_path', type=str, default='./saved_models/index_nyctaxi_30')
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--early_stop_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--display_steps', type=int, default=100)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--kernel', type=int, default=4)
    parser.add_argument('--time', type=str, default='30')
    parser.add_argument('--interval', type=int, default=48)
    parser.add_argument('--name', type=str, default=False)
    args = parser.parse_args()


    # path
    if args.name==False:
        args.output_dir = "./saved_models/index_{}_{}".format(args.index,args.time)
    else:
        args.output_dir ="./saved_models/{}".format(args.name)
    # os.system("mkdir -p {}".format(args.output_dir))  
    if args.pretrained_model_path != False:
        args.output_dir=args.output_dir+'_pretrained'
    os.makedirs(args.output_dir,exist_ok=True)
    print(args.output_dir)

    #log
    logger = logging.getLogger('train')
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    handler = logging.FileHandler(os.path.join(args.output_dir, "train_log.txt"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s -   %(message)s')  
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("Argument %s", args)




    #load external factors
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

    #load traffic flow video
    video_scale = MinMaxScaler(feature_range=(-1, 1))
    external_scale = MinMaxScaler(feature_range=(-1, 1))
    X=np.load('../../data_{}/data_x_{}.npy'.format(file_name, args.time), allow_pickle=True)
    Y=torch.tensor(np.load('../../data_{}/data_y_{}.npy'.format(file_name, args.time), allow_pickle=True), dtype=torch.float32)

    #split train,val,test datasets
    min_index = 4 * 7 * args.interval
    val_min_index = 4 * 7 * args.interval
    train_min_index = 5 * 7 * args.interval
    b, f, c, h, w = X.shape

    #min-max scale
    external[min_index : -1 * train_min_index,0,0:2]=external_scale.fit_transform(external[min_index : -1 * train_min_index, 0, 0:2])
    external[-1 * train_min_index : -1 * val_min_index, 0, 0:2] = external_scale.transform(external[-1 * train_min_index : -1 * val_min_index, 0, 0:2])
    external[ -1 * val_min_index:, 0, 0:2] = external_scale.transform(external[ -1 * val_min_index:, 0, 0:2])
    external = torch.tensor(external, dtype=torch.float32)

    # min-max scale
    X = rearrange(X, 'b f c h w -> b (f c h w)')
    X[0 : -1 * train_min_index]= video_scale.fit_transform(X[0: -1 * train_min_index]) #train datasets
    X[-1 * train_min_index : -1 * val_min_index, ]=video_scale.transform(X[-1 * train_min_index: -1 * val_min_index, ])#val datasets
    X[-1 * val_min_index:, ] = video_scale.transform(X[-1 * val_min_index:, ])#test datasets
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
    test_dataloader = DataLoader(test_dataset,  batch_size=args.train_batch_size,
                                  num_workers=0)


    # models
    model = models.ctrNet(args,logger)

    #reload paramaters
    if args.pretrained_model_path !=False:
        dic=torch.load(os.path.join(args.pretrained_model_path, "pytorch_model.bin"))
        model.model.load_state_dict(dic,strict=True)
        model.model.eval()
        with torch.no_grad():
            pres, y = model.infer(val_dataloader)
            mse, mae = model.eval(pres, y)
            logger.info("  " + "*" * 20)
            logger.info('reload model from %s ',os.path.join(args.output_dir, "pytorch_model.bin"))
            logger.info(" dev Best loss : %s,%s", round(float(mse) , 4),round(float(mae),4))
            logger.info("  " + "*" * 20)
            reload_loss = mse
    else:
        reload_loss=None

    # train model
    model.train(train_dataloader, val_dataloader,reload_loss)

    # output
    model.reload()
    model.model.eval()
    with torch.no_grad():
        pres, y = model.infer(test_dataloader)
        mse, mae = model.eval(pres, y)
        logger.info(" Test loss : %s,%s", round(float(np.sqrt(mse)), 4), round(float(mae), 4))






