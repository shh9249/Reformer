import copy
import random
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
import torch.distributed as dist
import logging
import re
import pdb
import json
import numpy as np
from transformers import T5Tokenizer


class BaseDataset(Dataset):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path,self.dataset)
        self.phase=args.phase
        self.max_his_len = args.max_his_len
        self.his_sep = args.his_sep
        self.index_file = args.index_file
        self.add_prefix = args.add_prefix
        self.ft = args.ft
        self.post = args.post
        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None


    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)

    def get_new_tokens(self):

        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens

    def get_all_items(self):

        if self.all_items is not None:
            return self.all_items
        warm_items = np.load(os.path.join(self.data_path,"phase%s"%(self.phase), "warm_item.npy"), allow_pickle=True).tolist()
        cold_items = np.load(os.path.join(self.data_path,"phase%s"%(self.phase), "cold_item.npy"), allow_pickle=True).tolist()
        self.all_items = set()
        for i_id in warm_items:
            self.all_items.add("".join(self.indices[str(i_id)]))
        for i_id in cold_items:
            self.all_items.add("".join(self.indices[str(i_id)]))

        return self.all_items

    def get_all_items_v2(self):
        if self.all_items is not None:
            return self.all_items

        self.all_items = []
        for index in self.indices.values():
            self.all_items.append("".join(index))

        return self.all_items       

    def get_warm_items(self):
        self.warm_items = set()
        warm_items = np.load(os.path.join(self.data_path,"phase%s"%(self.phase), "warm_item.npy"), allow_pickle=True).tolist()
        self.warm_ids = warm_items
        for i_id in warm_items:
            self.warm_items.add("".join(self.indices[str(i_id)]))
        return self.warm_items

    def get_newold_warm_items(self):
        self.new_warm_items = set()
        self.old_warm_items = set()
        old_warm_items = np.load(os.path.join(self.data_path,"phase%s"%(self.phase-1), "warm_item.npy"), allow_pickle=True).tolist()
        new_warm_items = np.load(os.path.join(self.data_path,"phase%s"%(self.phase-1), "cold_item.npy"), allow_pickle=True).tolist()
        self.old_warm_ids = old_warm_items
        self.new_warm_ids = new_warm_items
        print("old/new",len(old_warm_items),len(new_warm_items))
        for i_id in old_warm_items:
            self.old_warm_items.add("".join(self.indices[str(i_id)]))
        for i_id in new_warm_items:
            self.new_warm_items.add("".join(self.indices[str(i_id)]))
        return self.old_warm_items,self.new_warm_items

    def get_cold_items(self):
        self.cold_items = set()
        cold_items = np.load(os.path.join(self.data_path,"phase%s"%(self.phase), "cold_item.npy"), allow_pickle=True).tolist()
        self.cold_ids = cold_items
        for i_id in cold_items:
            self.cold_items.add("".join(self.indices[str(i_id)]))
        return self.cold_items
    def get_prefix_allowed_tokens_fn(self, tokenizer):


        if self.allowed_tokens is None:
            self.allowed_tokens = {}
            for index in self.indices.values():
                for i, token in enumerate(index):
                    token_id = tokenizer(token)["input_ids"][0]
                    if i not in self.allowed_tokens.keys():
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)
            self.allowed_tokens[len(self.allowed_tokens.keys())] = set([tokenizer.eos_token_id])
        sep = [0]


        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            reversed_sent = sentence[::-1]
            for i in range(len(reversed_sent)):
                if reversed_sent[i:i + len(sep)] == sep[::-1]:
                    return list(self.allowed_tokens[i])

        return prefix_allowed_tokens_fn

    def _process_data(self):

        raise NotImplementedError



class SeqRecDataset(BaseDataset):
        
    def __init__(self, args, mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)
        self.items=None
        self.mode = mode
        self.prompt_id = prompt_id
        self.sample_num = sample_num
        self.prompt = "What would user be likely to purchase next after buying items {history} ?"
        # load data
        self._load_data()

        self._remap_items()
        
        # load data
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.inter_data = self._process_valid_data()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        elif self.mode == "test_warm":
            self.warm_items = self.get_warm_items()
            self.inter_data = self._process_test_warm_data()

        elif self.mode == "test_old_warm":
            self.old_warm_items,self.new_warm_items=self.get_newold_warm_items()
            self.inter_data = self._process_test_old_warm_data()

        elif self.mode == "test_new_warm":
            self.old_warm_items,self.new_warm_items=self.get_newold_warm_items()
            self.inter_data = self._process_test_new_warm_data()

        elif self.mode == "test_cold":
            self.cold_items = self.get_cold_items()
            self.inter_data = self._process_test_cold_data()
        else:
            raise NotImplementedError


    def get_items(self):
        if self.items is not None:
            return self.items

        self.items = set()
        if self.mode == "test_cold":
            item_set = self.cold_item
        elif self.mode == "test_hot":
            item_set = self.hot_item
        print("hot",len(list(self.hot_item)))
        for iid in list(item_set):
            self.items.add("".join(self.indices[str(iid)]))
        # for iid,index in zip(list(item_set),self.indices.values()):
        #     self.all_items.add("".join(index))

        return self.items
    def _load_data(self):
        self.warm_data = {}
        self.train_data = {}
        print(self.phase)

        for phase_idx in range(self.phase):
            
            train_data = np.load(os.path.join(self.data_path,"phase%s"%(phase_idx), "training_dict.npy"), allow_pickle=True).item()
            for uid in train_data:
                if uid not in self.warm_data:
                    self.warm_data[uid] = []
                self.warm_data[uid].extend(train_data[uid])

            valid_data = np.load(os.path.join(self.data_path,"phase%s"%(phase_idx), "validation_dict.npy"), allow_pickle=True).item()
            for uid in valid_data:
                if uid not in self.warm_data:
                    self.warm_data[uid] = []
                if len(valid_data[uid]):
                    self.warm_data[uid].extend(valid_data[uid])
    
        train_data = np.load(os.path.join(self.data_path,"phase%s"%(self.phase), "training_dict.npy"), allow_pickle=True).item()
        for uid in train_data:
            if uid not in self.train_data:
                self.train_data[uid] = []
            self.train_data[uid].extend(train_data[uid])
        if self.phase==0:
            for uid in self.train_data:
                self.warm_data[uid] = []
        if not self.ft:
            for uid in self.warm_data:
                self.train_data[uid] = self.warm_data[uid]+self.train_data[uid]
                self.warm_data[uid] = []
        print(self.train_data[3])
        cnt =0
        for uid in self.train_data:
            for iid in self.train_data[uid]:
                cnt+=1
        print(f"train {cnt}")

        cnt =0
        for uid in self.warm_data:
            for iid in self.warm_data[uid]:
                cnt+=1
        print(f"warm {cnt}")
            
        self.valid_data = np.load(os.path.join(self.data_path,"phase%s"%(self.phase), "validation_dict.npy"), allow_pickle=True).item()
        
        for uid in self.valid_data:
            for iid in self.valid_data[uid]:
                cnt+=1
        print(f"valid {cnt}")
        
        self.test_data = np.load(os.path.join(self.data_path,"phase%s"%(self.phase), "testing_dict.npy"), allow_pickle=True).item()
        
        if self.post and  "warmtrain" in self.post:
            print("warmtrain")
            with open(os.path.join(self.data_path, "phase%s"%(0),self.dataset + self.index_file), 'r') as f:
                self.indices = json.load(f)
        else:
            with open(os.path.join(self.data_path, "phase%s"%(self.phase),self.dataset + self.index_file), 'r') as f:
                self.indices = json.load(f)

    def _remap_items(self):

        self.remapped_warm = dict()
        for uid, items in self.warm_data.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_warm[uid] = new_items

        self.remapped_train = dict()
        for uid, items in self.train_data.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_train[uid] = new_items
        
        self.remapped_valid = dict()
        for uid, items in self.valid_data.items():
            new_items = ["".join(self.indices[str(i)]) for i in items] if len(items) else []
            self.remapped_valid[uid] = new_items

        self.remapped_test = dict()
        for uid, items in self.test_data.items():
            new_items = ["".join(self.indices[str(i)]) for i in items] if len(items) else []
            self.remapped_test[uid] = new_items


    def _process_train_data(self):

        inter_data = []
        for uid  in self.remapped_train:
            warm_items = self.remapped_warm[uid]
            items = self.remapped_train[uid]
            for i in range(0, len(items)):
                one_data = dict()
                if (len(warm_items)>1 and i==0) or i>=1:
                    
                    one_data["item"] = items[i]
                    history = warm_items+items[:i]
                    if self.max_his_len > 0:
                        history = history[-self.max_his_len:]
                    if self.add_prefix:
                        history = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(history)]
                    one_data["inters"] = self.prompt.format(history=",".join(history))
                    inter_data.append(one_data)

        return inter_data
    
    def _process_valid_data(self):

        inter_data = []
        for uid in self.remapped_valid:
            items = self.remapped_valid[uid]
            train_items = self.remapped_train[uid]
            warm_items = self.remapped_warm[uid]
            if len(items):
                one_data = dict()
                one_data["item"] = items[0]
                history = warm_items+train_items
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
                one_data["inters"] = self.prompt.format(history=",".join(history))
                inter_data.append(one_data)

        return inter_data    


    def _process_test_data(self):

        inter_data = []
        for uid in self.remapped_test:
            items = self.remapped_test[uid]
            train_items = self.remapped_train[uid]
            warm_items = self.remapped_warm[uid]
            valid_items = self.remapped_valid[uid]
            if len(items):
                one_data = dict()
                one_data["item"] = items
                history = warm_items+train_items+valid_items
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
                one_data["inters"] = self.prompt.format(dataset=self.dataset,history=",".join(history))
                inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def _process_test_old_warm_data(self):
        warm_cnt = 0
        inter_data = []
        for uid in self.remapped_test:
            items = self.remapped_test[uid]
            train_items = self.remapped_train[uid]
            warm_items = self.remapped_warm[uid]
            valid_items = self.remapped_valid[uid]
            ids = self.test_data[uid]
            if len(items):
                one_data = dict()
                gold = []
                for iid,item in zip(ids,items):
                    if iid in self.old_warm_ids:
                        gold.append(item)
                        warm_cnt += 1
                one_data["item"] = gold
                history = warm_items+train_items + valid_items
                if len(gold):
                    if self.max_his_len > 0:
                        history = history[-self.max_his_len:]
                    if self.add_prefix:
                        history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
                    one_data["inters"] = self.prompt.format(dataset=self.dataset,history=",".join(history))
                    inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()
        print("old warm interaction in test:", warm_cnt)
        return inter_data  

    def _process_test_new_warm_data(self):
        warm_cnt = 0
        inter_data = []
        for uid in self.remapped_test:
            items = self.remapped_test[uid]
            train_items = self.remapped_train[uid]
            warm_items = self.remapped_warm[uid]
            valid_items = self.remapped_valid[uid]
            ids = self.test_data[uid]
            if len(items):
                one_data = dict()
                gold = []
                for iid,item in zip(ids,items):
                    if iid in self.new_warm_ids:
                        gold.append(item)
                        warm_cnt += 1
                one_data["item"] = gold
                history = warm_items+train_items + valid_items
                if len(gold):
                    if self.max_his_len > 0:
                        history = history[-self.max_his_len:]
                    if self.add_prefix:
                        history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
                    one_data["inters"] = self.prompt.format(dataset=self.dataset,history=",".join(history))
                    inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()
        print("new warm interaction in test:", warm_cnt)
        return inter_data  

    def _process_test_warm_data(self):
        warm_cnt = 0
        inter_data = []
        for uid in self.remapped_test:
            items = self.remapped_test[uid]
            train_items = self.remapped_train[uid]
            warm_items = self.remapped_warm[uid]
            valid_items = self.remapped_valid[uid]
            ids = self.test_data[uid]
            if len(items):
                one_data = dict()
                gold = []
                for iid,item in zip(ids,items):
                    if iid in self.warm_ids:
                        gold.append(item)
                        warm_cnt += 1
                one_data["item"] = gold
                history = warm_items+train_items + valid_items
                if len(gold):
                    if self.max_his_len > 0:
                        history = history[-self.max_his_len:]
                    if self.add_prefix:
                        history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
                    one_data["inters"] = self.prompt.format(dataset=self.dataset,history=",".join(history))
                    inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()
        print("warm interaction in test:", warm_cnt)
        return inter_data
    

    def _process_test_cold_data(self):
        inter_data = []
        cold_cnt = 0
        for uid in self.remapped_test:
            items = self.remapped_test[uid]
            train_items = self.remapped_train[uid]
            warm_items = self.remapped_warm[uid]
            valid_items = self.remapped_valid[uid]
            ids = self.test_data[uid]
            if len(items):
                one_data = dict()
                gold = []
                for iid,item in zip(ids,items):
                    if iid in self.cold_ids: #TODO
                        gold.append(item)
                        cold_cnt += 1
                one_data["item"] = gold
                history = warm_items+train_items + valid_items
                if len(gold):
                    if self.max_his_len > 0:
                        history = history[-self.max_his_len:]
                    if self.add_prefix:
                        history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
                    one_data["inters"] = self.prompt.format(dataset=self.dataset,history=",".join(history))
                    inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()
        print("cold interaction in test:", cold_cnt)
        return inter_data

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):

        return len(self.inter_data)

    def __getitem__(self, index):


        d = self.inter_data[index]


        return dict(input_ids=d["inters"], labels=d["item"])


    
