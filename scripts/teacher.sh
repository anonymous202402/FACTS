export CUDA_VISIBLE_DEVICES=0

model_name=MTS_31F
seq_len=48

python -u run_teacher.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /you_data_path  \
  --data_path /your_data_file_path \
  --model_id Folsom_$seq_len'_'24 \
  --model $model_name \
  --data Folsom \
  --features M \
  --seq_len $seq_len \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 42 \
  --dec_in 42 \
  --c_out 42 \
  --des 'Exp' \
  --tors teacher \
  --itr 3
