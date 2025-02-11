import argparse
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import sys
from typing import List, Dict

import torch

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(gpu_name)

import transformers
from peft import PeftModel 
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Qwen2Tokenizer, Qwen2Config, Qwen2ForCausalLM
from utils import *
from collator import TestCollator
from evaluate import get_topk_results, get_metrics_results
from generation_trie import Trie

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

import ipdb
from test_qwen1 import Trie, prefix_allowed_tokens_fn, get_greedy_prefix_allowed_tokens_fn, get_topk_results

def test(args):

    set_seed(args.seed)
    print(vars(args))

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        print(vars(args))

    dist.init_process_group(backend="nccl", world_size=world_size, rank=local_rank)

    device_map = {"": local_rank}
    device = torch.device("cuda",local_rank)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.ckpt_path)
    tokenizer.padding_side = "left"

    load_8bit = True 
    #load_8bit = False
    dtype = torch.bfloat16 
    bf16 = True 

    if not args.lora:
        args.base_model = args.ckpt_path

    model = Qwen2ForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        load_in_8bit=load_8bit,
        device_map=device_map,
    )
    model.resize_token_embeddings(len(tokenizer))

    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        modules_to_save=args.lora_modules_to_save.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)
    if args.ckpt_path:
        checkpoint_name = os.path.join(
            args.ckpt_path, "adapter_model.bin"
        )  
        args.ckpt_path = False  
        if os.path.exists(checkpoint_name):
        
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
            del adapters_weights
        else:
            if local_rank == 0:
                print(f"Checkpoint {checkpoint_name} not found")
                
    model = DistributedDataParallel(model, device_ids=[local_rank])
    model.eval()
    prompt_ids = [0]

    test_data = load_test_dataset(args)
    if args.subset_test:
        args.sample_num = 200
        test_data = load_test_dataset(args)
    ddp_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=local_rank, shuffle=False, drop_last=False)

    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()

    indices = test_data.indices
    warm_items = np.load(os.path.join(args.data_path,args.dataset,"phase%s"%(args.phase), "warm_item.npy"), allow_pickle=True).tolist()
    cold_items = np.load(os.path.join(args.data_path,args.dataset,"phase%s"%(args.phase), "cold_item.npy"), allow_pickle=True).tolist()
    col_dict = {}
    for _ in  warm_items:
        _ = "".join(indices[str(_)])
        if _ not in col_dict:
            col_dict[_]=0
        col_dict[_]+=1
    for _ in  cold_items:
        _ = "".join(indices[str(_)])
        if _ not in col_dict:
            col_dict[_]=0
        col_dict[_]+=1
    all_len = len(warm_items)+len(cold_items)
    print("col rate",len(col_dict),(all_len-len(col_dict))/all_len)


    if args.greedy_trie:
        prefix_allowed_tokens = get_greedy_prefix_allowed_tokens_fn(test_data.indices,tokenizer)
    else:
        candidate_trie = Trie(
                [
                    [1] + 
                    tokenizer.encode(candidate)
                    + [tokenizer.eos_token_id]
                    for candidate in all_items
                ]
            )
        prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie, tokenizer)
    

    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                              num_workers=2, pin_memory=True,shuffle=False)#,

    if local_rank == 0:
        print("data num:", len(test_data))
        

    

    import time 
    # all performance
    with torch.no_grad(): 
        for prompt_id in prompt_ids:

            test_loader.dataset.set_prompt(prompt_id)
            total = 0

            all_pred_list = []
            all_gold_list = []
            
            st_all = time.time()
            local_all_pred_list = []
            local_all_gold_list = []
            for step, batch in enumerate(tqdm(test_loader)):
                inputs = batch[0].to(device)
                targets = batch[1]
                if step%world_size!=local_rank:
                    continue
                total += len(targets)
                output = model.module.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=args.max_new_token, 
                    prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                    do_sample=False
                )
                
                output_ids = output["sequences"]
                scores = output["sequences_scores"]
                output = tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )

                topk_res = get_topk_results(output,scores,targets,args.num_beams,
                                            all_items=all_items if args.filter_items else None)
                local_all_pred_list.extend(topk_res)
                local_all_gold_list.extend(targets)
            dist.barrier()
            res_gather_list = [None for _ in range(world_size)]
            dist.all_gather_object(obj=local_all_pred_list, object_list=res_gather_list)
            target_gather_list = [None for _ in range(world_size)]
            dist.all_gather_object(obj=local_all_gold_list, object_list=target_gather_list)
            
            if local_rank == 0:
                for ga_res in res_gather_list:
                    all_pred_list.extend(ga_res)

                for ga_tar in target_gather_list:
                    all_gold_list.extend(ga_tar)
            if local_rank == 0:
                print("=== End ===%s"%(local_rank))
                test_results = computeTopNAccuracy(all_gold_list, all_pred_list, topN=[5, 10, 20],rank=local_rank,col_dict = col_dict)
                print_results(None, None, test_results)
        dist.barrier()
            # print results
        
        if local_rank == 0:
            test_results = computeTopNAccuracy(all_gold_list, all_pred_list, topN=[5, 10, 20],col_dict = col_dict)
            print("=== End ===")
            print("=== All performance")
            print_results(None, None, test_results)
            print(f"All time costs: {round(time.time()-st_all, 2)}s")
            
    