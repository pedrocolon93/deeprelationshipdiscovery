python ../code/adversarial.py \
    --seen_file ../data/ft_nb_seen_ook.txt \
    --adjusted_file ../data/nb_retrofitted_ook_attractrepel.txt  \
    --unseen_file ../data/ft_nb_unseen_ook.txt \
    --out_dir ../results/numberbatch_ook-n_epochs=1-adam_lr=0.001/ \
    --n_epochs 1 \
    --map_optimizer adam,lr=0.001 \
    --dis_optimizer adam,lr=0.001 \
    --sim_optimizer adam,lr=0.001
