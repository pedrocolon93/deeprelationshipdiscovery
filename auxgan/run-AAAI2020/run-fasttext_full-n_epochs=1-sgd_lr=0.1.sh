python ../code/adversarial.py \
    --seen_file ../data/fasttext_seen.txt \
    --adjusted_file ../data/fasttext_seen_attractrepelretrofitted.txt \
    --unseen_file ../data/fasttext_seen.txt \
    --out_dir ../results/fasttext_full-n_epochs=1-sgd_lr=0.1/ \
    --n_epochs 1 \
    --map_optimizer sgd,lr=0.1 \
    --dis_optimizer sgd,lr=0.1 \
    --sim_optimizer sgd,lr=0.1
