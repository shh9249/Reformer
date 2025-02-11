dataset=
history_index=
file_name=
python generate_indices-warmtrain.py \
    --all_phase 5 \
    --phase 0 \
    --dataset $dataset \
    --ckpt_path ckpt/$dataset/freq_best_collision_model.pth\
    --postfix $file_name\



python generate_indices_append.py \
    --phase 1 \
    --dataset $dataset \
    --ckpt_path ckpt/$dataset/freq_best_collision_model.pth\
    --postfix $file_name \
    --pre_indices $history_index
