python ../code/adversarial.py \
    --seen_file ../data/glove_seen.txt \
    --adjusted_file ../data/glove_seen_attractrepelretrofitted.txt \
    --unseen_file ../data/glove_seen.txt \
    --out_dir ../results/glove_full-n_epochs=10-sgd_lr=0.1/ \
    --n_epochs 10 \
    --map_optimizer sgd,lr=0.1 \
    --dis_optimizer sgd,lr=0.1 \
    --sim_optimizer sgd,lr=0.1
