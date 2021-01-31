export CUDA_VISIBLE_DEVICES=7

##############
##          ##
## Training ##
##          ##
##############

python ../code/adversarial.py \
    --seen_file ../data/ft_nb_seen_ook.txt \
    --adjusted_file ../data/nb_retrofitted_ook_attractrepel.txt  \
    --unseen_file ../data/ft_nb_unseen_ook_plus_card.txt \
    --out_dir ../results-ACL2021/numberbatch_ook-n_epochs=10-adam_lr=0.1/ \
    --n_epochs 10 \
    --map_optimizer adam,lr=0.1 \
    --dis_optimizer adam,lr=0.1 \
    --sim_optimizer adam,lr=0.1 \
    --in_simlex_file ../evaluation/simlexorig999.txt \
    --in_simverb_file ../evaluation/simverb3500.txt \
    --in_card_file ../evaluation/card660.tsv \
    --in_embedding_file ../results-ACL2021/numberbatch_ook-n_epochs=10-adam_lr=0.1/gold_embs.txt
