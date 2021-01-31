python ../code/adversarial.py \
    --seen_file ../data/fasttext_seen.txt \
    --adjusted_file ../data/fasttext_seen_attractrepelretrofitted.txt \
    --unseen_file ../data/fasttext_seen.txt \
    --out_dir ../results/fasttext_full-n_epochs=10-adam_lr=0.1/ \
    --n_epochs 10 \
    --map_optimizer adam,lr=0.1 \
    --dis_optimizer adam,lr=0.1 \
    --sim_optimizer adam,lr=0.1
