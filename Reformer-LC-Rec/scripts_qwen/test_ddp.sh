export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1 

lora="--lora"
subset=
DATASET=
model_class=
ckpt=
phase=
index_file=
gpu_ids=(4 5 6 7)

for lr in 3e-4
    do
        (
        CKPT_PATH=
        logfile=../../log/Qwen/test/${DATASET}/phase$phase/Reformer-LC-Rec-Qwen2.5-1.5B-${lr}lr-0wd-lora-qvoud-64r-128a-bf16-int8-log.txt 

        CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node=4 --use_env ../../test_qwen_ddp.py \
            --dataset $DATASET \
            --base_model /path_to_model/Qwen2.5-1.5B \
            --ckpt_path $CKPT_PATH \
            --test_batch_size  12\
            --num_beams 20 \
            --lora_r 64 \
            --lora_modules_to_save "embed_tokens,lm_head" \
            --lora_alpha 128 \
            --index_file $index_file \
            ${lora} \
            --phase $phase \
            --ft 0 \
            --lora \
            &> ${logfile}
        )
    done

