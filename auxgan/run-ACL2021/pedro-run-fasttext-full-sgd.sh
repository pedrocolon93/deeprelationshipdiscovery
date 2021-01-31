export CUDA_VISIBLE_DEVICES=6

##############
##          ##
## Training ##
##          ##
##############

python ../code/adversarial.py \
    --seen_file ../data/fixpedro_ft_all_unseen_fasttext_full.txt \
    --adjusted_file ../data/fixpedro_ft_all_unseen_retrofitted.txt \
    --unseen_file ../data/fixpedro_ft_all_unseen_fasttext_full.txt \
    --out_dir ../results-ACL2021/pedro_fasttext_full_sgd/ \
    --n_epochs 10 \
    --map_optimizer sgd,lr=0.1 \
    --dis_optimizer sgd,lr=0.1 \
    --sim_optimizer sgd,lr=0.1 \
    --in_simlex_file ../evaluation/simlexorig999.txt \
    --in_simverb_file ../evaluation/simverb3500.txt \
    --in_card_file ../evaluation/card660.tsv \
    --in_embedding_file ../results-ACL2021/pedro_fasttext_full_sgd/gold_embs.txt
