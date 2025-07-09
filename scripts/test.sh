export CUDA_VISIBLE_DEVICES=0

model_name=MTS
model_img_name=MTS_3
seq_len=48

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --kd_type response \
  --kd_loss_weight 0.1 \
  --root_path /opt/data/private/code/code/PCA/data_csv/  \
  --data_path 3_merged_features_weather.csv \
  --model_id Folsom_$seq_len'_'24 \
  --model $model_name \
  --model_img $model_img_name \
  --data Folsom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 42 \
  --dec_in 42 \
  --c_out 42 \
  --des 'Exp' \
  --itr 1


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --kd_type causal \
#   --kd_loss_weight 0.1 \
#   --root_path /opt/data/private/code/code/PCA/data_csv/  \
#   --data_path merged_features_block_2dpca.csv \
#   --model_id Folsom_$seq_len'_'24 \
#   --model $model_name \
#   --data Folsom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 42 \
#   --dec_in 42 \
#   --c_out 42 \
#   --des 'Exp' \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --kd_type contrastive \
#   --kd_loss_weight 0.01 \
#   --root_path /opt/data/private/code/code/PCA/data_csv/  \
#   --data_path merged_features_block_2dpca.csv \
#   --model_id Folsom_$seq_len'_'24 \
#   --model $model_name \
#   --data Folsom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 42 \
#   --dec_in 42 \
#   --c_out 42 \
#   --des 'Exp' \
#   --itr 1


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --kd_type causal \
#   --kd_loss_weight 0.01 \
#   --root_path /opt/data/private/code/code/PCA/data_csv/  \
#   --data_path merged_features_block_2dpca.csv \
#   --model_id Folsom_$seq_len'_'24 \
#   --model $model_name \
#   --data Folsom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 42 \
#   --dec_in 42 \
#   --c_out 42 \
#   --des 'Exp' \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --kd_type response \
#   --kd_loss_weight 1.0 \
#   --root_path /opt/data/private/code/code/PCA/data_csv/  \
#   --data_path merged_features_block_2dpca.csv \
#   --model_id Folsom_$seq_len'_'24 \
#   --model $model_name \
#   --data Folsom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 42 \
#   --dec_in 42 \
#   --c_out 42 \
#   --des 'Exp' \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --kd_type relation \
#   --kd_loss_weight 1.0 \
#   --root_path /opt/data/private/code/code/PCA/data_csv/  \
#   --data_path merged_features_block_2dpca.csv \
#   --model_id Folsom_$seq_len'_'24 \
#   --model $model_name \
#   --data Folsom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 42 \
#   --dec_in 42 \
#   --c_out 42 \
#   --des 'Exp' \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --kd_type contrastive \
#   --kd_loss_weight 1.0 \
#   --root_path /opt/data/private/code/code/PCA/data_csv/  \
#   --data_path merged_features_block_2dpca.csv \
#   --model_id Folsom_$seq_len'_'24 \
#   --model $model_name \
#   --data Folsom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 42 \
#   --dec_in 42 \
#   --c_out 42 \
#   --des 'Exp' \
#   --itr 1


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --kd_type causal \
#   --kd_loss_weight 1.0 \
#   --root_path /opt/data/private/code/code/PCA/data_csv/  \
#   --data_path merged_features_block_2dpca.csv \
#   --model_id Folsom_$seq_len'_'24 \
#   --model $model_name \
#   --data Folsom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 42 \
#   --dec_in 42 \
#   --c_out 42 \
#   --des 'Exp' \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --kd_type response \
#   --kd_loss_weight 0.1 \
#   --root_path /opt/data/private/code/code/PCA/data_csv/  \
#   --data_path merged_features_block_2dpca.csv \
#   --model_id Folsom_$seq_len'_'24 \
#   --model $model_name \
#   --data Folsom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 42 \
#   --dec_in 42 \
#   --c_out 42 \
#   --des 'Exp' \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --kd_type relation \
#   --kd_loss_weight 0.1 \
#   --root_path /opt/data/private/code/code/PCA/data_csv/  \
#   --data_path merged_features_block_2dpca.csv \
#   --model_id Folsom_$seq_len'_'24 \
#   --model $model_name \
#   --data Folsom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 42 \
#   --dec_in 42 \
#   --c_out 42 \
#   --des 'Exp' \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --kd_type contrastive \
#   --kd_loss_weight 0.1 \
#   --root_path /opt/data/private/code/code/PCA/data_csv/  \
#   --data_path merged_features_block_2dpca.csv \
#   --model_id Folsom_$seq_len'_'24 \
#   --model $model_name \
#   --data Folsom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 42 \
#   --dec_in 42 \
#   --c_out 42 \
#   --des 'Exp' \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --kd_type causal \
#   --kd_loss_weight 0.1 \
#   --root_path /opt/data/private/code/code/PCA/data_csv/  \
#   --data_path merged_features_block_2dpca.csv \
#   --model_id Folsom_$seq_len'_'24 \
#   --model $model_name \
#   --data Folsom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 42 \
#   --dec_in 42 \
#   --c_out 42 \
#   --des 'Exp' \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --kd_type response \
#   --kd_loss_weight 0.01 \
#   --root_path /opt/data/private/code/code/PCA/data_csv/  \
#   --data_path merged_features_block_2dpca.csv \
#   --model_id Folsom_$seq_len'_'24 \
#   --model $model_name \
#   --data Folsom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 42 \
#   --dec_in 42 \
#   --c_out 42 \
#   --des 'Exp' \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --kd_type relation \
#   --kd_loss_weight 0.01 \
#   --root_path /opt/data/private/code/code/PCA/data_csv/  \
#   --data_path merged_features_block_2dpca.csv \
#   --model_id Folsom_$seq_len'_'24 \
#   --model $model_name \
#   --data Folsom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 42 \
#   --dec_in 42 \
#   --c_out 42 \
#   --des 'Exp' \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --kd_type contrastive \
#   --kd_loss_weight 0.01 \
#   --root_path /opt/data/private/code/code/PCA/data_csv/  \
#   --data_path merged_features_block_2dpca.csv \
#   --model_id Folsom_$seq_len'_'24 \
#   --model $model_name \
#   --data Folsom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 42 \
#   --dec_in 42 \
#   --c_out 42 \
#   --des 'Exp' \
#   --itr 1


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --kd_type causal \
#   --kd_loss_weight 0.01 \
#   --root_path /opt/data/private/code/code/PCA/data_csv/  \
#   --data_path merged_features_block_2dpca.csv \
#   --model_id Folsom_$seq_len'_'24 \
#   --model $model_name \
#   --data Folsom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 42 \
#   --dec_in 42 \
#   --c_out 42 \
#   --des 'Exp' \
#   --itr 1