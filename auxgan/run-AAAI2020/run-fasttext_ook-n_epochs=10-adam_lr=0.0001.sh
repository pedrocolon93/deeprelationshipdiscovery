python ../code/adversarial.py \
    --seen_file ../data/fasttext_seen_ook.txt \
    --adjusted_file ../data/fasttext_seen_ook_attractrepelretrofitted.txt \
    --unseen_file ../data/fasttext_seen_ook.txt \
    --out_dir ../results/fasttext_ook-n_epochs=10-adam_lr=0.0001/ \
    --n_epochs 10 \
    --map_optimizer adam,lr=0.0001 \
    --dis_optimizer adam,lr=0.0001 \
    --sim_optimizer adam,lr=0.0001
