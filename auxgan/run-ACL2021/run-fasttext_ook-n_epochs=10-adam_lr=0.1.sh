export CUDA_VISIBLE_DEVICES=6

##############
##          ##
## Training ##
##          ##
##############

python ../code/adversarial.py \
    --seen_file ../data/fasttext_seen_ook.txt \
    --adjusted_file ../data/fasttext_seen_ook_attractrepelretrofitted.txt \
    --unseen_file ../data/fasttext_seen_plus_card.txt \
    --out_dir ../results-ACL2021/fasttext_ook-n_epochs=10-adam_lr=0.1/ \
    --n_epochs 10 \
    --map_optimizer adam,lr=0.1 \
    --dis_optimizer adam,lr=0.1 \
    --sim_optimizer adam,lr=0.1 \
    --in_simlex_file ../evaluation/simlexorig999.txt \
    --in_simverb_file ../evaluation/simverb3500.txt \
    --in_card_file ../evaluation/card660.tsv \
    --in_embedding_file ../results-ACL2021/fasttext_ook-n_epochs=10-adam_lr=0.1/gold_embs.txt
