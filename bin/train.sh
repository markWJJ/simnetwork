python trainer.py --model esim \
 --model_config /notebooks/source/simnet/model_config.json \
 --model_dir /data/xuht/merged_data_model/simnet \
 --config_prefix /notebooks/source/simnet/configs \
 --gpu_id 2 \
 --train_path "/data/xuht/duplicate_sentence/merged_data/train.txt" \
 --dev_path "/data/xuht/duplicate_sentence/merged_data/dev.txt" \
 --w2v_path "/data/xuht/Chinese_w2v/sgns.merge.char/sgns.merge.char.pkl" \
 --vocab_path "/data/xuht/duplicate_sentence/merged_data/emb_mat.pkl"

