import argparse
import random
import torch
import numpy as np
from time import time
import logging
import os 
from torch.utils.data import DataLoader

from datasets import EmbDataset, new_EmbDataset
from models.rqvae import RQVAE
from trainer_test import  Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, )
    parser.add_argument('--eval_step', type=int, default=50, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument("--data_path", type=str,
                        default="",
                        help="Input data path.")

    parser.add_argument("--warm_codebook",type=str,default=None)

    parser.add_argument('--weight_decay', type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--init", type=str, default="kmeans", help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0, 0.003], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")
    #not used
    parser.add_argument("--replace_freq", type=int, default=0, help="replace")
    parser.add_argument('--affine_lr', type=float,default=0.0, help="sinkhorn epsilons")
    parser.add_argument("--affine_groups", type=int, default=1, help="max sinkhorn iters")
    parser.add_argument("--freq_policy", type=str, default=None, help="max sinkhorn iters")
    parser.add_argument("--push_freq", type=int, default=50, help="max sinkhorn iters")
    parser.add_argument("--push_start", type=int, default=1000, help="max sinkhorn iters")
    parser.add_argument("--push_lr", type=int, default=1e-4, help="max sinkhorn iters")

    parser.add_argument("--device", type=str, default="cuda:1", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256,256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument('--layers', type=int, nargs='+', default=[2048,1024,512,256,128,64], help='hidden sizes of every layer')

    parser.add_argument("--ckpt_dir", type=str, default="", help="output directory for model")
    parser.add_argument("--phase", type=int, default=0, help="output directory for model")
    parser.add_argument("--dataset", type=str, default=None, help="output directory for model")
    parser.add_argument("--a",type=float, nargs='+', default=[0.1, 0.1, 0.1, 0.1], help="max sinkhorn iters")
    parser.add_argument("--new_a",type=float, nargs='+', default=[1, 1, 1, 1], help="max sinkhorn iters")
    parser.add_argument("--b",type=float, nargs='+', default=[0.0, 0.0, 0.0, 0.0], help="max sinkhorn iters")
    parser.add_argument("--b_scale",type=float, nargs='+', default=[0.0, 0.0, 0.0, 0.0], help="max sinkhorn iters")
    parser.add_argument("--iso",type=int, default=0)
    parser.add_argument("--seed",type=int, default=2023)
    

    return parser.parse_args()


def worker_init_fn(worked_id):
    worker_seed = 2023
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == '__main__':
    """fix the random seed"""
    args = parse_args()
    print(args)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    

    logging.basicConfig(level=logging.DEBUG)

    """build dataset"""
    warm_data = []
    for _ in range(args.phase):
        if _==0:
            warm_data.append(EmbDataset(args.data_path,_,args.dataset))
        else:
            warm_data.append(new_EmbDataset(args.data_path,_-1,args.dataset))
    if args.phase!=0:
        data = new_EmbDataset(args.data_path,args.phase-1,args.dataset)
    else:
        data = EmbDataset(args.data_path,0,args.dataset)
    model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  init=args.init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  affine_lr=args.affine_lr,
                  affine_groups=args.affine_groups,
                  replace_freq=args.replace_freq,
                  a = args.a,
                  new_a = args.new_a,
                  b = args.b,
                  b_scale = args.b_scale,
                  freq_policy=args.freq_policy,
                  warm_codebook = args.warm_codebook,
                  iso=args.iso,
                  device=args.device,
                  seed=args.seed
                  )
    import math
    sep_r = np.argmin([abs(math.ceil(l/i)-args.batch_size) for i in range(1,11)])
    args.batch_size = math.ceil(l/(sep_r+1))
    print("batch size",args.batch_size)
    if len(warm_data):
        warm_data_loader = [DataLoader(_,num_workers=args.num_workers,
                                batch_size=args.batch_size, shuffle=False,
                                pin_memory=True) for _ in warm_data]
    else:
        warm_data_loader = None
    data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)
    trainer = Trainer(args,model)
    best_loss, best_collision_rate = trainer.fit(warm_data_loader,data_loader)

    print("Best Loss",best_loss)
    print("Best Collision Rate", best_collision_rate)

