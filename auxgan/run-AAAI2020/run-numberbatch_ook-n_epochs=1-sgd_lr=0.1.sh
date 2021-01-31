python ../code/adversarial.py \
    --seen_file ../data/ft_nb_seen_ook.txt \
    --adjusted_file ../data/nb_retrofitted_ook_attractrepel.txt  \
    --unseen_file ../data/ft_nb_unseen_ook.txt \
    --out_dir ../results/numberbatch_ook-n_epochs=1-sgd_lr=0.1/ \
    --n_epochs 1 \
    --map_optimizer sgd,lr=0.1 \
    --dis_optimizer sgd,lr=0.1 \
    --sim_optimizer sgd,lr=0.1
