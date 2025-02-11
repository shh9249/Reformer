dataset=
seed=
python main.py \
  --device cuda:1\
  --data_path /your_path/Reformer-TIGER/data/$dataset \
  --ckpt_dir /your_path/Reformer-TIGER/RQ-VAE/ckpt/$dataset/path \
  --phase  0\
  --dataset $dataset \
  --epoch 20000\
  --a 0.5 0 0 0  \
  --new_a 0.5 0 0 0  \
  --freq_policy pow \
  --sk_epsilons 0.0 0.0 0.003 \
  --num_emb_list 64 64 64\
  --lr 1e-3 \
  --seed $seed
