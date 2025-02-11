dataset=
seed=
ckpt_path=
python main.py \
  --device cuda:4\
  --data_path /your_path/Reformer-TIGER/data/$dataset \
  --ckpt_dir $ckpt_path\
  --phase  0\
  --dataset $dataset \
  --epoch 20000\
  --a 0.25 0 0 0  \
  --new_a 0.25 0 0 0  \
  --freq_policy pow \
  --sk_epsilons 0.0 0.0 0.003 \
  --num_emb_list 64 64 64\
  --lr 1e-3 \
  --seed $seed
