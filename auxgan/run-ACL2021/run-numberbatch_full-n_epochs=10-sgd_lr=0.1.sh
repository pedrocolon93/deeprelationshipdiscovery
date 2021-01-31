export CUDA_VISIBLE_DEVICES=5

##############
##          ##
## Training ##
##          ##
##############

python ../code/adversarial.py \
    --seen_file ../data/ft_nb_seen.txt \
    --adjusted_file ../data/nb_retrofitted_attractrepel.txt \
    --unseen_file ../data/ft_nb_seen_plus_card.txt \
    --out_dir ../results-ACL2021/numberbatch_full-n_epochs=10-sgd_lr=0.1/ \
    --n_epochs 10 \
    --map_optimizer sgd,lr=0.1 \
    --dis_optimizer sgd,lr=0.1 \
    --sim_optimizer sgd,lr=0.1 \
    --in_simlex_file ../evaluation/simlexorig999.txt \
    --in_simverb_file ../evaluation/simverb3500.txt \
    --in_card_file ../evaluation/card660.tsv \
    --in_embedding_file ../results-ACL2021/numberbatch_full-n_epochs=10-sgd_lr=0.1/gold_embs.txt
