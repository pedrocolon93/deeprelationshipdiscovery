python ../code/adversarial.py \
    --seen_file ../data/ft_nb_seen.txt \
    --adjusted_file ../data/nb_retrofitted_attractrepel.txt \
    --unseen_file ../data/ft_nb_seen.txt \
    --out_dir ../results/numberbatch_full-n_epochs=10-adam_lr=0.001/ \
    --n_epochs 10 \
    --map_optimizer adam,lr=0.001 \
    --dis_optimizer adam,lr=0.001 \
    --sim_optimizer adam,lr=0.001
