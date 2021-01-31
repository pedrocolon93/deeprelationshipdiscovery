python ../code/adversarial.py \
    --seen_file ../data/glove_seen_ook.txt \
    --adjusted_file ../data/glove_seen_ook_attractrepelretrofitted.txt \
    --unseen_file ../data/glove_seen_ook.txt \
    --out_dir ../results/glove_ook-n_epochs=1-sgd_lr=0.1/ \
    --n_epochs 1 \
    --map_optimizer sgd,lr=0.1 \
    --dis_optimizer sgd,lr=0.1 \
    --sim_optimizer sgd,lr=0.1
