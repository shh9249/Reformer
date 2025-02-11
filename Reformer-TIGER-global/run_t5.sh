export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=2334 ./finetune_pret.py \
    --output_dir \  
    --phase  0\        #period
    --dataset \        #your dataset  
    --per_device_batch_size 320 \
    --learning_rate 5e-3 \
    --epochs 800 \
    --index_file  \    #identifier file
    --base_model \     #model path
    --seed 42 \
    --ft  \            #fine-tuning or retraining
    --ckpt_path        #resume from checkpoint


logfile=logfile_name
python test_col.py \
    --gpu_id \
    --ckpt_path  \      #test model
    --dataset  \        #your dataset
    --test_batch_size 120 \
    --num_beams 20 \
    --index_file  \     #identifier file
    --phase 1 \
    --ft 0 \
    &> ${logfile}&

