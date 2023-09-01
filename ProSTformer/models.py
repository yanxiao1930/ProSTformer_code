import torch
import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import prostformer as Prostformer
from torch.utils.data import DataLoader,SequentialSampler, RandomSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,get_constant_schedule_with_warmup,
                          RobertaConfig, RobertaModel)
import random
from sklearn.metrics import mean_absolute_error
from einops import rearrange, repeat


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

class ctrNet(nn.Module):
    def __init__(self,args,logger):
        super().__init__()
        comfig = {
            # 现在表上的最好超参数！！！！！
            'nyctaxi': {
                'dim': 32,
                'dim_1':32*12,
                'exter_dim': 56,
                'num_frames': 4,  # 4
                'periods': 3,  # 3
                'image_size': (12, 16),
                'patch_size': (1,1),
                'patch_size_1': (3,4),
                'channels': 2,
                'depth':6,
                'heads': 8,
                's':16,
                'n':12,
                'heads_1': 8,
                'dim_head': 4,
                'dim_head_1':48,
                'attn_dropout': args.dropout,
                'ff_dropout': args.dropout
            },

            'nycbike': {
                'dim': 32,
                'dim_1': 32*6,
                'exter_dim': 56,
                'num_frames': 4,  # 4
                'periods': 3,  # 3
                'image_size': (12, 16),
                'patch_size': (1,1),
                'patch_size_1': (3,2),
                'channels': 2,
                'depth':6,
                'heads': 8,
                's':32,
                'n':6,
                'heads_1': 8,
                'dim_head': 4,
                'dim_head_1':24,
                'attn_dropout': args.dropout,
                'ff_dropout': args.dropout
            }

        }


        #设置GPU和创建模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.device = device
        logger.info(" device: %s, n_gpu: %s",device, args.n_gpu)
        param=comfig[args.index]
        logger.info("model config = %s", param)
        Model= Prostformer.TimeSformer(
            dim=param['dim'],
            dim_1=param['dim_1'],
            exter_dim=param['exter_dim'],
            num_frames=param['num_frames'],  # 4
            periods=param['periods'],  # 3
            image_size=param['image_size'],
            patch_size=param['patch_size'],
            patch_size_1=param['patch_size_1'],
            channels=param['channels'],
            depth=param['depth'],
            heads=param['heads'],
            heads_1=param['heads_1'],
            dim_head=param['dim_head'],
            dim_head_1=param['dim_head_1'],
            attn_dropout=param['attn_dropout'],
            ff_dropout=param['ff_dropout'],
            s=param['s'],
            n=param['n']
        )
        model = Model
        self.model=model.to(args.device)
        self.args = args
        self.logger=logger
        set_seed(args)

    def train(self, train_dataloader,dev_dataloader=None,best_loss=None):#train_dataset是TensorDataset(x1,x2,y)
        args = self.args
        logger=self.logger
        stop_epochs=args.early_stop_epoch#早停步数
        early_stop_lis=np.zeros(stop_epochs)
        # 设置dataloader
        # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size,
        #                               num_workers=0)
        # dev_dataloader = DataLoader(dev_dataset, shuffle=True, batch_size=args.train_batch_size,
        #                               num_workers=0)
        args.max_steps = args.epoch * len(train_dataloader)
        args.save_steps = len(train_dataloader) // 10
        args.warmup_steps = len(train_dataloader)
        args.logging_steps = len(train_dataloader)
        args.num_train_epochs = args.epoch
        # 设置优化器
        optimizer = AdamW(self.model.parameters(), lr=args.lr, eps=1e-8, weight_decay=0.08)
        #warm_up
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(
        len(train_dataloader) * args.num_train_epochs * 0.2), num_training_steps=int(
        len(train_dataloader) * args.num_train_epochs))
        # 多GPU设置
        if args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        model = self.model
        # 开始训练
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataloader))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        if args.n_gpu != 0:
            logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // args.n_gpu)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.train_batch_size)
        logger.info("  Total optimization steps = %d", args.max_steps)

        global_step = 0
        tr_loss, avg_loss, tr_nb = 0.0,  0.0, 0.0
        best_loss=float('inf') if best_loss==None else best_loss#恢复模型的话计算best_loss
        model.zero_grad()
        # scheduler.step(240*len(train_dataloader))
        patience = 0
        for idx in range(args.num_train_epochs):
            tr_num = 0
            train_loss = 0
            stop_index = idx % stop_epochs
            for step, batch in enumerate(train_dataloader):
                # forward和backward
                video,external,y= [x.to(args.device) for x in batch]
                del batch
                model.train()#启用batch normalization和drop out。model.eval()，这时神经网络会沿用batch normalization的值，并不使用drop out。
                loss,_ = model(video,external,y)
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                loss.backward()#计算梯度
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)#梯度裁剪
                tr_loss += loss.item()
                tr_num += 1
                train_loss += loss.item()
                # 输出log
                if avg_loss == 0:
                    avg_loss = tr_loss
                avg_loss = round(train_loss / tr_num, 5)
                if (step + 1) % args.display_steps == 0:
                    logger.info("  epoch {} step {} loss {}".format(idx, step + 1, avg_loss))
                # update梯度
                optimizer.step()#更新梯度
                optimizer.zero_grad()#重置梯度
                scheduler.step()
                global_step += 1

            # 一个epoch结束后，测试验证集结果
            if dev_dataloader is not None:
                #infer的这种慢，20s对100s
                # dev_video=dev_dataset[0].to(args.device)
                # dev_externals = dev_dataset[1].to(args.device)
                # dev_y = dev_dataset[2].to(args.device)
                model.eval()
                # mse,mae = model(dev_video, dev_externals, dev_y)
                pres, y = self.infer(dev_dataloader)
                mse, mae = self.eval(pres, y)

                # 输出loss
                results = {}
                results['eval_loss'] = mse
                results['eval_mae'] = mae
                # 打印结果
                for key, value in results.items():
                    logger.info(" epoch %s, %s = %s", idx, key, round(float(value), 4))
                # 保存最好的loss和模型
                if  results['eval_loss'] < best_loss:
                    early_stop_lis[stop_index] = 1
                    best_loss = results['eval_loss']
                    logger.info("  " + "*" * 20)
                    logger.info("  Best loss : %s", round(float(best_loss), 4))
                    logger.info("  " + "*" * 20)
                    try:
                        os.system("mkdir -p {}".format(args.output_dir))#创建目录
                    except:
                        pass
                    model_to_save = model.module if hasattr(model,'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                #early_stop_policy
                else:
                    early_stop_lis[stop_index] = -1
                if len(early_stop_lis) == -sum(early_stop_lis):
                    logger.info("eraly stop no changed (%s)", args.early_stop_epoch)
                    return
                else:
                    pass

    def infer(self,dev_dataloader):
        #预测年龄和性别的概率分布
        args=self.args
        model=self.model
        pre_lis=[]
        y_lis=[]
        model.eval()
        for batch in dev_dataloader:
            dev_video, dev_external, dev_y = [x.to(args.device) for x in batch]
            del batch
            with torch.no_grad():
                pre = model(dev_video, dev_external)
            pre_lis.append(pre.cpu())
            y_lis.append(dev_y.cpu())

        age_probs=torch.cat(pre_lis,0)
        dev_y=torch.cat(y_lis,0)
        return age_probs,dev_y

    # def eval(self,preds,labels):
    #     #求出loss和acc
    #     loss=F.mse_loss(preds,labels).mean().item()
    #     return loss

    def eval(self,preds,truth):
        def MAE(pred, gt):
            mae = torch.abs(pred - gt).mean()
            return mae
        #求出loss和acc
        loss = F.mse_loss(preds, truth)
        mae = MAE(preds, truth)
        # loss=F.mse_loss(preds,truth).mean().item()
        # pre_reshape = rearrange(preds, ' f c h w -> ( f c h w)')
        # test_reshape = rearrange(truth, ' f c h w -> ( f c h w)')
        # mae = mean_absolute_error(test_reshape, pre_reshape)
        return loss,mae


    def reload(self):
        # 读取在验证集结果最好的模型
        model = self.model
        args = self.args
        logger=self.logger
        args.load_model_path = os.path.join(args.output_dir, "pytorch_model.bin")
        logger.info("Load model from %s", args.load_model_path)
        model_to_load = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        model_to_load.load_state_dict(torch.load(args.load_model_path))


