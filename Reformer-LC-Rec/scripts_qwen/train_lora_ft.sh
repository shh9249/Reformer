export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1 
DATASET=Software_11111_0.1
lora="--lora"
only_train_response="--only_train_response"
model_class=Qwen2.5-1.5B
subset=""
ft=1
ckpt_name=
index_name=
post_name=
lr=
seed=
for wd in 0
do
    suffix=lora-qvoud-64r-128a-bf16-int8-${seed}-seed
    suffix=${model_class}-${lr}lr-${wd}wd-${suffix}
    logfile=../../log/Qwen//${DATASET}/phase${phase}/${index_name}/train_${suffix}-log.txt 
    logdir=../../log/Qwen//${DATASET}/phase${phase}/${index_name}
    if [ ! -d "$logdir" ]; then
        mkdir -p "$logdir"
        echo "Directory $logdir created."
    else
        echo "Directory $logdir already exists."
    fi
    OUTPUT_DIR=../../ckpt/Qwen/${DATASET}/phase${phase}/${index_name}/${suffix}
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=5880 ../finetune_lora.py \
        --base_model /path_to_model/${model_class} \
        --output_dir $OUTPUT_DIR \
        --subseq \
        --dataset $DATASET \
        --per_device_batch_size 16 \
        --gradient_accumulation_steps 2 \
        --learning_rate $lr \
        --epochs 50 \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_target_modules "q_proj,v_proj,o_proj,up_proj,down_proj" \
        --weight_decay $wd \
        --save_and_eval_strategy steps \
        --warmup_steps 200 \
        --lora_modules_to_save "embed_tokens,lm_head" \
        --index_file ${index_name} \
        --special_token_for_answer "|start_of_answer|" \
        --test_batch_size 4 \
        --resume_from_checkpoint ${ckpt_name} \
        --num_beams 20 \
        --phase ${phase} \
        --ft 1 \
        --post $post_name \
        --seed ${seed} \
        ${subset} \
        ${only_train_response} \
        &> ${logfile}
        wait
done

