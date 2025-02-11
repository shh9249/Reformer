import argparse
import json
import os
import sys
from typing import List
import math
import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from utils import *
from collator import TestCollator
from evaluate import get_topk_results, get_metrics_results
from generation_trie import Trie



def test(args):

    set_seed(args.seed)
    print(vars(args))

    device_map = {"": args.gpu_id}
    device = torch.device("cuda",args.gpu_id)

    tokenizer = T5Tokenizer.from_pretrained(args.ckpt_path)
    model = T5ForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )

    prompt_ids = [0]

    test_data = load_test_dataset(args)


    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()
    print("all",len(all_items))
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
    candidate_trie = Trie(
        [
            [0] + tokenizer.encode(candidate)
            for candidate in all_items
        ]
    )
    prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                             shuffle=False, num_workers=0, pin_memory=True)



    model.eval()


    import time 
    
    # all performance
    with torch.no_grad(): 
        for prompt_id in prompt_ids:

            test_loader.dataset.set_prompt(prompt_id)
            total = 0

            all_pred_list = []
            all_gold_list = []
            st_all = time.time()
            for step, batch in enumerate(tqdm(test_loader)):
                inputs = batch[0].to(device)
                targets = batch[1]
                total += len(targets)

                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=10,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                )

                output_ids = output["sequences"]
                scores = output["sequences_scores"]
                output = tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )

                topk_res = get_topk_results(output,scores,targets,args.num_beams,
                                            all_items=all_items if args.filter_items else None)
                all_pred_list.extend(topk_res)
                all_gold_list.extend(targets)
                
            test_results = computeTopNAccuracy(all_gold_list, all_pred_list, topN=[5, 10, 20],col_dict = col_dict)
            print("=== End ===")
            print("=== All performance")
            print_results(None, None, test_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()
    test(args)