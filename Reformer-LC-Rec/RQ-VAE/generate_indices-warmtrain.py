import collections
import json
import logging
import argparse
import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import warm_TokenDataset,cold_TokenDataset,sp_TokenDataset
from models.rqvae import RQVAE

import os
def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE")

    parser.add_argument('--all_phase', type=int, default=5, help='learning rate')
    parser.add_argument('--iso', type=int, default=0, help='learning rate')
    parser.add_argument('--phase', type=int, default=0, help='learning rate')
    parser.add_argument('--dataset', type=str, default=None, help='learning rate')
    parser.add_argument('--ckpt_path', type=str, default=None, help='learning rate')
    parser.add_argument('--postfix', type=str, default=None, help='learning rate')

    return parser.parse_args()

def check_collision(warm_all_indices_str):
    tot_item = len(warm_all_indices_str)
    tot_indice = len(set(warm_all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(warm_all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in warm_all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(warm_all_indices_str):
    index2id = {}
    for i, index in enumerate(warm_all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups


pargs = parse_args()
phase=pargs.phase
iso = pargs.iso
dataset = pargs.dataset
ckpt_path = pargs.ckpt_path
output_dir = 
output_file = f"{dataset}{pargs.postfix}.json"
output_emb_file = f"{dataset}{pargs.postfix}.pt"
output_emb_file = os.path.join(output_dir,output_emb_file)
output_file = os.path.join(output_dir,output_file)
device = torch.device("cuda:7")
data_path = 
ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
args = ckpt["args"]
state_dict = ckpt["state_dict"]
warm_data = warm_TokenDataset(data_path,phase,dataset)
cold_data = cold_TokenDataset(data_path,phase,dataset)

model = RQVAE(in_dim=warm_data.dim,
                  num_emb_list=[ [_] for _ in args.num_emb_list],
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
                  affine_lr=0.0,
                  affine_groups=1,
                  replace_freq=0,
                  a = args.a,
                  new_a = args.a,
                  b = args.b,
                  b_scale = args.b_scale,
                  freq_policy=args.freq_policy,
                  )

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print(model)
freq=args.a
print(freq,args.freq_policy)
all_weights=[]
warm_data_loader = DataLoader(warm_data,num_workers=args.num_workers,
                             batch_size=1024, shuffle=False,
                             pin_memory=True)

cold_data_loader = DataLoader(cold_data,num_workers=args.num_workers,
                             batch_size=1024, shuffle=False,
                             pin_memory=True)

warm_all_indices = []
warm_all_indices_str = []
all_emb = []
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]
first_assign = set()
if freq:
    _dist_scale=torch.ones(model.rq.vq_layers[0].codebook[0].weight.shape[0]).to(device)
    _bias = torch.zeros(model.rq.vq_layers[0].codebook[0].weight.shape[0]).to(device)
for d in tqdm(warm_data_loader):
    d = d.to(device)
    indices,emb= model.get_indices_emb(d,use_sk=False,scale=_dist_scale)
    for _ in indices:
        first_assign.add(int(_[0].detach().cpu()))
    all_emb.append(emb)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = []
        for i, ind in enumerate(index):
            code.append(prefix[i].format(int(ind)))

        warm_all_indices.append(code)
        warm_all_indices_str.append(str(code))

warm_all_indices = np.array(warm_all_indices)
warm_all_indices_str = np.array(warm_all_indices_str)
for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon=0.0

if model.rq.vq_layers[-1].sk_epsilon == 0.0:
    model.rq.vq_layers[-1].sk_epsilon = 0.003

tt = 0
while True:
    if tt >= 11 or check_collision(warm_all_indices_str):
        break
    collision_item_groups = get_collision_item(warm_all_indices_str)
    for collision_items in collision_item_groups:
        d = warm_data[collision_items].to(device)
        indices = model.get_indices(d, use_sk=True,scale=_dist_scale)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for item, index in zip(collision_items, indices):
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))

            warm_all_indices[item] = code
            warm_all_indices_str[item] = str(code)
    tt += 1
import copy
print("All indices number: ",len(warm_all_indices))
print("Max number of conflicts: ", max(get_indices_count(warm_all_indices_str).values()))
tot_item = len(warm_all_indices_str)
tot_indice = len(set(warm_all_indices_str.tolist()))
print("Collision Rate",(tot_item-tot_indice)/tot_item)

all_indices_dict = {}
for item, indices in enumerate(warm_all_indices.tolist()):
    all_indices_dict[item] = list(indices)


cold_all_indices = []
cold_all_indices_str = []
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]

for d in tqdm(cold_data_loader):
    d = d.to(device)
    indices,emb= model.get_indices_emb(d,use_sk=False,scale=_dist_scale)
    all_emb.append(emb)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = []
        for i, ind in enumerate(index):
            code.append(prefix[i].format(int(ind)))

        cold_all_indices.append(code)
        cold_all_indices_str.append(str(code))
cold_all_indices = np.array(cold_all_indices)
cold_all_indices_str = np.array(cold_all_indices_str)

for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon=0.0

if model.rq.vq_layers[-1].sk_epsilon == 0.0:
    model.rq.vq_layers[-1].sk_epsilon = 0.003

tt = 0
while True:
    if tt >= 11 or check_collision(cold_all_indices_str):
        break
    collision_item_groups = get_collision_item(cold_all_indices_str)
    for collision_items in collision_item_groups:
        d = cold_data[collision_items].to(device)
        indices = model.get_indices(d, use_sk=True,scale=_dist_scale)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for item, index in zip(collision_items, indices):
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))

            cold_all_indices[item] = code
            cold_all_indices_str[item] = str(code)
    tt += 1
print("All indices number: ",len(cold_all_indices))
print("Max number of conflicts: ", max(get_indices_count(cold_all_indices_str).values()))

tot_item = len(cold_all_indices_str)
tot_indice = len(set(cold_all_indices_str.tolist()))
print("Collision Rate",(tot_item-tot_indice)/tot_item)

for item, indices in enumerate(cold_all_indices.tolist()):
    item = item + len(warm_data)
    if item in all_indices_dict:
        print(item)
        exit()
    all_indices_dict[item] = list(indices)

def get_sp(sp_data_loader,sp_data,start_item,scale):
    sp_all_indices = []
    sp_all_indices_str = []
    prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]

    for d in tqdm(sp_data_loader):
        d = d.to(device)
        indices,emb= model.get_indices_emb(d,use_sk=False,scale=scale)
        all_emb.append(emb)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))

            sp_all_indices.append(code)
            sp_all_indices_str.append(str(code))
    sp_all_indices = np.array(sp_all_indices)
    sp_all_indices_str = np.array(sp_all_indices_str)

    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon=0.0

    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003

    tt = 0
    while True:
        if tt >= 11 or check_collision(sp_all_indices_str):
            break
        collision_item_groups = get_collision_item(sp_all_indices_str)
        for collision_items in collision_item_groups:
            d = sp_data[collision_items].to(device)
            indices = model.get_indices(d, use_sk=True,scale=_dist_scale)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = []
                for i, ind in enumerate(index):
                    code.append(prefix[i].format(int(ind)))

                sp_all_indices[item] = code
                sp_all_indices_str[item] = str(code)
        tt += 1
    print("All indices number: ",len(sp_all_indices))
    print("Max number of conflicts: ", max(get_indices_count(sp_all_indices_str).values()))
    
    tot_item = len(sp_all_indices_str)
    tot_indice = len(set(sp_all_indices_str.tolist()))
    print("Collision Rate",(tot_item-tot_indice)/tot_item)
    for item, indices in enumerate(sp_all_indices.tolist()):
        item = item + start_item
        if item in all_indices_dict:
            print(item)
            exit()
        all_indices_dict[item] = list(indices)

start_item = len(warm_data)+len(cold_data)
for _ in range(phase+1,pargs.all_phase-1):
    sp_data = sp_TokenDataset(data_path,_,dataset)
    sp_data_loader = DataLoader(sp_data,num_workers=args.num_workers,
                                batch_size=1024, shuffle=False,
                                pin_memory=True)
    get_sp(sp_data_loader,sp_data,start_item,_dist_scale)
    start_item+=len(sp_data)

with open(output_file, 'w') as fp:
    json.dump(all_indices_dict,fp)
all_emb = torch.cat(all_emb, 0)
print(all_emb.size())
with open(output_emb_file, 'w') as fp:
    torch.save(all_emb, output_emb_file)
    print(all_emb[0])