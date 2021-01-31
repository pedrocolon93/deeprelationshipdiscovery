python ../code/adversarial.py \
    --seen_file ../data/ft_nb_seen.txt \
    --adjusted_file ../data/nb_retrofitted_attractrepel.txt \
    --unseen_file ../data/ft_nb_seen.txt \
    --out_dir ../results/numberbatch_full-n_epochs=10-sgd_lr=0.1/ \
    --n_epochs 10 \
    --map_optimizer sgd,lr=0.1 \
    --dis_optimizer sgd,lr=0.1 \
    --sim_optimizer sgd,lr=0.1
