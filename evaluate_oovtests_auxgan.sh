# Directory: /home/yxin/research/projects/RetroGAN-DRD/retrogandeeprelationshipdiscovery

### First, post-specialize the gold embeddings
#python ../auxgan/code/export.py \
#--params $OOVTEST_DIR/params.pkl \
#--model $OOVTEST_DIR/best_mapping.t7 \
#--in_file $OOVTEST_DIR/gold_embs.txt \
#--out_file $OOVTEST_DIR/postspec_gold_embs.txt

## Evaluate with SimLex-999
function evaluate_simlex() {
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
    python ../auxgan/evaluation/simlex_evaluator.py \
    ../auxgan/evaluation/simlexorig999.txt \
    $OOVTEST_DIR/gold_embs.txt
    echo "Evaluated simlex for $OOVTEST_DIR" > "evaluated_simlex_$OOVTEST_DIR.txt"
}

## Evaluate with SimVerb-3500
function evaluate_simverb() {
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
    python ../auxgan/evaluation/simlex_evaluator.py \
    ../auxgan/evaluation/simverb3500.txt \
    $OOVTEST_DIR/gold_embs.txt
    echo "Evaluated simverb for $OOVTEST_DIR" > "evaluated_simverb_$OOVTEST_DIR.txt"
}
