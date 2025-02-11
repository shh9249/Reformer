import collections
import json
import logging

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader

from datasets import warm_TokenDataset,cold_TokenDataset
from models.rqvae import RQVAE

import os
def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE")

    parser.add_argument('--iso', type=int, default=0, help='learning rate')
    parser.add_argument('--phase', type=int, default=0, help='learning rate')
    parser.add_argument('--dataset', type=str, default=None, help='learning rate')
    parser.add_argument('--ckpt_path', type=str, default=None, help='learning rate')
    parser.add_argument('--postfix', type=str, default=None, help='learning rate')
    parser.add_argument('--pre_indices', type=str, default=None, help='learning rate')
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


phase=pargs.phase
iso = pargs.iso
dataset = pargs.dataset
ckpt_path = pargs.ckpt_path

output_dir = 
pre_indice = pargs.pre_indices
output_file = f"{dataset}{pargs.postfix}.json"
output_file = os.path.join(output_dir,output_file)
device = torch.device("cuda:7")

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
args = ckpt["args"]
state_dict = ckpt["state_dict"]
emb_list = [64]
for _ in range(phase):
    emb_list.append(16)
num_emb_list=[emb_list,emb_list,emb_list]
warm_data = warm_TokenDataset(args.data_path,phase-1,dataset)
append_data = cold_TokenDataset(args.data_path,phase-1,dataset)
cold_data = cold_TokenDataset(args.data_path,phase,dataset)
model = RQVAE(in_dim=warm_data.dim,
                  num_emb_list=num_emb_list,
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
                  a = args.a,
                  new_a = args.new_a,
                  b = args.b,
                  b_scale = args.b_scale,
                  freq_policy=args.freq_policy,
                  iso=iso
                  )

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
freq = np.sum(args.a) or np.sum(args.new_a)
_dist_scale=torch.ones(np.sum(num_emb_list[0])).to(device)
warm_data_loader = DataLoader(warm_data,num_workers=args.num_workers,
                             batch_size=64, shuffle=False,
                             pin_memory=True)
append_data_loader = DataLoader(append_data,num_workers=args.num_workers,
                             batch_size=64, shuffle=False,
                             pin_memory=True)
cold_data_loader = DataLoader(cold_data,num_workers=args.num_workers,
                             batch_size=64, shuffle=False,
                             pin_memory=True)
pre_indices_dict={}
with open(pre_indice,"r") as f:
    pre_indices_dict=json.load(f)

all_indices_dict = {}
for _ in range(len(warm_data)):
    _=str(_)
    all_indices_dict[_]=pre_indices_dict[_]

append_all_indices = []
append_all_indices_str = []
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]

for d in tqdm(append_data_loader):
    d = d.to(device)
    indices,emb= model.get_indices_emb(d,use_sk=False,scale=_dist_scale,p=phase)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = []
        for i, ind in enumerate(index):
            if i >=1  and iso:
                code.append(prefix[i].format(int(ind)+sum(emb_list[:-1])))
            else:
                code.append(prefix[i].format(int(ind)))

        append_all_indices.append(code)
        append_all_indices_str.append(str(code))
append_all_indices = np.array(append_all_indices)
append_all_indices_str = np.array(append_all_indices_str)

for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon=0.0

if model.rq.vq_layers[-1].sk_epsilon == 0.0:
    model.rq.vq_layers[-1].sk_epsilon = 0.003

tt = 0
while True:
    if tt >= 11 or check_collision(append_all_indices_str):
        break
    collision_item_groups = get_collision_item(append_all_indices_str)
    for collision_items in collision_item_groups:
        d = append_data[collision_items].to(device)
        indices = model.get_indices(d, use_sk=True,scale=_dist_scale,p=phase)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for item, index in zip(collision_items, indices):
            code = []
            for i, ind in enumerate(index):
                if i >=1  and iso:
                    code.append(prefix[i].format(int(ind)+sum(emb_list[:-1])))
                else:
                    code.append(prefix[i].format(int(ind)))

            append_all_indices[item] = code
            append_all_indices_str[item] = str(code)
    tt += 1

print("All indices number: ",len(append_all_indices))
print("Max number of conflicts: ", max(get_indices_count(append_all_indices_str).values()))

tot_item = len(append_all_indices_str)
tot_indice = len(set(append_all_indices_str.tolist()))
print("Collision Rate",(tot_item-tot_indice)/tot_item)

for item, indices in enumerate(append_all_indices.tolist()):
    item = item + len(warm_data)
    if item in all_indices_dict:
        print(item)
        exit()
    all_indices_dict[item] = list(indices)

cold_all_indices = []
cold_all_indices_str = []
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]

for d in tqdm(cold_data_loader):
    d = d.to(device)
    indices,emb= model.get_indices_emb(d,use_sk=False,scale=_dist_scale,p=phase)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = []
        for i, ind in enumerate(index):
            if i >=1  and iso:
                code.append(prefix[i].format(int(ind)+sum(emb_list[:-1])))
            else:
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
        indices = model.get_indices(d, use_sk=True,scale=_dist_scale,p=phase)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for item, index in zip(collision_items, indices):
            code = []
            for i, ind in enumerate(index):
                if i >=1  and iso:
                    code.append(prefix[i].format(int(ind)+sum(emb_list[:-1])))
                else:
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
    item = item + len(warm_data) + len(append_data)
    if item in all_indices_dict:
        print(item)
        exit()
    all_indices_dict[item] = list(indices)

with open(output_file, 'w') as fp:
    json.dump(all_indices_dict,fp)