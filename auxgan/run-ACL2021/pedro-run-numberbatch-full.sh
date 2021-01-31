export CUDA_VISIBLE_DEVICES=7

##############
##          ##
## Training ##
##          ##
##############

python ../code/adversarial.py \
    --seen_file ../data/fixpedro_ft_all_unseen_numberbatch_full.txt \
    --adjusted_file ../data/fixpedro_nb_retrofitted_attractrepelretrofitted.txt \
    --unseen_file ../data/fixpedro_ft_all_unseen_numberbatch_full.txt \
    --out_dir ../results-ACL2021/pedro_numberbatch_full/ \
    --n_epochs 10 \
    --map_optimizer adam,lr=0.1 \
    --dis_optimizer adam,lr=0.1 \
    --sim_optimizer adam,lr=0.1 \
    --in_simlex_file ../evaluation/simlexorig999.txt \
    --in_simverb_file ../evaluation/simverb3500.txt \
    --in_card_file ../evaluation/card660.tsv \
    --in_embedding_file ../results-ACL2021/pedro_numberbatch_full/gold_embs.txt
