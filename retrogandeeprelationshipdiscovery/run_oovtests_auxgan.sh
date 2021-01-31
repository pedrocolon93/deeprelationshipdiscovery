#PATH_TO_AR_PYTHON=/home/yxin/anaconda3/envs/.attractrepel/bin/python  # peterchin
#PATH_TO_AUXGAN_PYTHON=/home/yxin/anaconda3/envs/.auxgan/bin/python  # peterchin
#PATH_TO_AR="/home/yxin/research/projects/RetroGAN-DRD/attract-repel"  # peterchin
#ORIGINAL_VECTORS="/home/yxin/research/projects/RetroGAN-DRD/data/cc.en.300.vec"  # peterchin
PATH_TO_AR_PYTHON="/scratch/yxin/.yida_attractrepel/bin/python"  # peterchin5
PATH_TO_AUXGAN_PYTHON="/scratch/yxin/.yida_auxgan/bin/python"  # peterchin5
PATH_TO_AR="/scratch/yxin/RetroGAN-DRD/attract-repel"  # peterchin5
ORIGINAL_VECTORS="/scratch/yxin/RetroGAN-DRD/data/cc.en.300.vec"  # peterchin5
CARD_FT_VECS="/scratch/yxin/RetroGAN-DRD/auxgan/evaluation/card_ft_vecs.txt"
ARVECTOR_POSTFIXFILENAME="cc.en.300.ar.vec"

CURR_DIR=$(pwd)

## PERCENTAGE and LEARNING_RATE are both provided in actual bash files
#PERCENTAGEREP=${PERCENTAGE/\./_}
#LEARNING_RATE_REP=${LEARNING_RATE/\./_}
#
#OUTDIR="oovtest-$PERCENTAGEREP-$OPTIMIZER-lr-$LEARNING_RATE_REP"

echo "Working in"
echo $CURR_DIR

PERCENTAGE=0.05
SEED=42

##################################
##                              ##
## Generate data for percentage ##
##                              ##
##################################

function generate_data_for_percentage() {
    echo "Working in $CURR_DIR/$OUTDIR/"
    echo "Outputting AR TO:"
    echo "$CURR_DIR/$OUTDIR/ar$PERCENTAGEREP$ARVECTOR_POSTFIXFILENAME"
    mkdir $OUTDIR
    python oov_cutter_slsv.py --target_file simlexsimverb_words.txt --percentage_to_leave $PERCENTAGE --seed $SEED --output_dir "$OUTDIR/"
    CONSTRAINTS=synonyms.txt
    # python oov_cutter_slsv.py --target_file testing/SimLex-999.txt --percentage_to_leave $PERCENTAGE --seed $SEED --output_dir "$OUTDIR/"
    # python oov_cutter_slsv.py --target_file testing/SimVerb-3500.txt --percentage_to_leave $PERCENTAGE --seed $SEED --output_dir "$OUTDIR/"
    # python oov_cutter_slsv_constraints.py --seen_words "$OUTDIR/SimLex-999_cut_to_$PERCENTAGEREP.txt" --all_constraints $CONSTRAINTS --output_dir "$OUTDIR/"
    # python oov_cutter_slsv_constraints.py --seen_words "$OUTDIR/SimVerb-3500_cut_to_$PERCENTAGEREP.txt" --all_constraints $CONSTRAINTS --output_dir "$OUTDIR/"
    python oov_cutter_slsv_constraints.py --seen_words "$OUTDIR/simlexsimverb_words_cut_to_$PERCENTAGEREP.txt" --all_constraints $CONSTRAINTS --output_dir "$OUTDIR/"
    echo "Fusing both"
    # cat $OUTDIR/synonyms_reducedwith_SimLex-999_$PERCENTAGEREP.txt $OUTDIR/synonyms_reducedwith_SimVerb-3500_$PERCENTAGEREP.txt > $OUTDIR/synonyms_reducedwith_$PERCENTAGEREP.txt
    cp $OUTDIR/synonyms_reducedwith_simlexsimverb_$PERCENTAGEREP.txt $OUTDIR/synonyms_reducedwith_$PERCENTAGEREP.txt
    # python oov_cutter_slsv_constraints_removeoverlap.py  --simlexcut "$OUTDIR/synonyms_reducedwith_SimLex-999_$PERCENTAGEREP.txt" --simverbcut "$OUTDIR/synonyms_reducedwith_SimVerb-3500_$PERCENTAGEREP.txt" --outputfile "$OUTDIR/synonyms_reducedwith_$PERCENTAGEREP.txt"
    CONSTRAINTS=antonyms.txt
    # python oov_cutter_slsv.py --target_file testing/SimLex-999.txt --percentage_to_leave $PERCENTAGE --seed $SEED --output_dir "$OUTDIR/"
    # python oov_cutter_slsv.py --target_file testing/SimVerb-3500.txt --percentage_to_leave $PERCENTAGE --seed $SEED --output_dir "$OUTDIR/"
    # python oov_cutter_slsv_constraints.py --seen_words "$OUTDIR/SimLex-999_cut_to_$PERCENTAGEREP.txt" --all_constraints $CONSTRAINTS --output_dir "$OUTDIR/"
    # python oov_cutter_slsv_constraints.py --seen_words "$OUTDIR/SimVerb-3500_cut_to_$PERCENTAGEREP.txt" --all_constraints $CONSTRAINTS --output_dir "$OUTDIR/"
    python oov_cutter_slsv_constraints.py --seen_words "$OUTDIR/simlexsimverb_words_cut_to_$PERCENTAGEREP.txt" --all_constraints $CONSTRAINTS --output_dir "$OUTDIR/"
    echo "Fusing both"
    cp $OUTDIR/antonyms_reducedwith_simlexsimverb_$PERCENTAGEREP.txt $OUTDIR/antonyms_reducedwith_$PERCENTAGEREP.txt
    # cat $OUTDIR/antonyms_reducedwith_SimLex-999_$PERCENTAGEREP.txt $OUTDIR/antonyms_reducedwith_SimVerb-3500_$PERCENTAGEREP.txt > $OUTDIR/antonyms_reducedwith_$PERCENTAGEREP.txt
    # python oov_cutter_slsv_constraints_removeoverlap.py  --simlexcut "$OUTDIR/antonyms_reducedwith_SimLex-999_$PERCENTAGEREP.txt" --simverbcut "$OUTDIR/antonyms_reducedwith_SimVerb-3500_$PERCENTAGEREP.txt" --outputfile "$OUTDIR/antonyms_reducedwith_$PERCENTAGEREP.txt"
}

#function generate_data_for_percentage() {
#    echo "Working in $CURR_DIR/$OUTDIR/"
#    echo "Outputting AR TO:"
#    echo "$CURR_DIR/$OUTDIR/ar$PERCENTAGEREP$ARVECTOR_POSTFIXFILENAME"
#    mkdir $OUTDIR
#    CONSTRAINTS=synonyms.txt
#    python oov_cutter_slsv.py --target_file testing/SimLex-999.txt --percentage_to_leave $PERCENTAGE --seed $SEED --output_dir $OUTDIR
#    python oov_cutter_slsv.py --target_file testing/SimVerb-3500.txt --percentage_to_leave $PERCENTAGE --seed $SEED --output_dir $OUTDIR
#    python oov_cutter_slsv_constraints.py --seen_words "$OUTDIR/SimLex-999_cut_to_$PERCENTAGEREP.txt" --all_constraints $CONSTRAINTS --output_dir $OUTDIR
#    python oov_cutter_slsv_constraints.py --seen_words "$OUTDIR/SimVerb-3500_cut_to_$PERCENTAGEREP.txt" --all_constraints $CONSTRAINTS --output_dir $OUTDIR
#    echo "Fusing both"
#    cat $OUTDIR/synonyms_reducedwith_SimLex-999_$PERCENTAGEREP.txt "$OUTDIR/synonyms_reducedwith_SimVerb-3500_$PERCENTAGEREP.txt" > "$OUTDIR/synonyms_reducedwith_$PERCENTAGEREP.txt"
#    # python oov_cutter_slsv_constraints_removeoverlap.py  --simlexcut "$OUTDIR/synonyms_reducedwith_SimLex-999_$PERCENTAGEREP.txt" --simverbcut "$OUTDIR/synonyms_reducedwith_SimVerb-3500_$PERCENTAGEREP.txt" --outputfile "$OUTDIR/synonyms_reducedwith_$PERCENTAGEREP.txt"
#    CONSTRAINTS=antonyms.txt
#    python oov_cutter_slsv.py --target_file testing/SimLex-999.txt --percentage_to_leave $PERCENTAGE --seed $SEED --output_dir $OUTDIR
#    python oov_cutter_slsv.py --target_file testing/SimVerb-3500.txt --percentage_to_leave $PERCENTAGE --seed $SEED --output_dir $OUTDIR
#    python oov_cutter_slsv_constraints.py --seen_words $OUTDIR/SimLex-999_cut_to_$PERCENTAGEREP.txt --all_constraints $CONSTRAINTS --output_dir $OUTDIR
#    python oov_cutter_slsv_constraints.py --seen_words $OUTDIR/SimVerb-3500_cut_to_$PERCENTAGEREP.txt --all_constraints $CONSTRAINTS --output_dir $OUTDIR
#    echo "Fusing both"
#    cat $OUTDIR/antonyms_reducedwith_SimLex-999_$PERCENTAGEREP.txt $OUTDIR/antonyms_reducedwith_SimVerb-3500_$PERCENTAGEREP.txt > $OUTDIR/antonyms_reducedwith_$PERCENTAGEREP.txt
#    # python oov_cutter_slsv_constraints_removeoverlap.py  --simlexcut "$OUTDIR/antonyms_reducedwith_SimLex-999_$PERCENTAGEREP.txt" --simverbcut "$OUTDIR/antonyms_reducedwith_SimVerb-3500_$PERCENTAGEREP.txt" --outputfile "$OUTDIR/antonyms_reducedwith_$PERCENTAGEREP.txt"
#}

##################################
##                              ##
## attract-repel for percentage ##
##                              ##
##################################

function attractrepel_for_percentage() {  # this function adds prefix "en_" to words
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
    python data_prep_retrogan.py --arconfigname "arconfig_$PERCENTAGE.config" --path_to_ar $PATH_TO_AR \
    --path_to_ar_python $PATH_TO_AR_PYTHON --synonyms "$OUTDIR/synonyms_reducedwith_$PERCENTAGEREP.txt" \
    --antonyms "$OUTDIR/antonyms_reducedwith_$PERCENTAGEREP.txt" --ccn $ORIGINAL_VECTORS --aroutput "$CURR_DIR/$OUTDIR/ar$PERCENTAGEREP$ARVECTOR_POSTFIXFILENAME" \
    --output_dir $OUTDIR
}

###############################
##                           ##
## run-AAAI2020 AuxGAN for percentage ##
##                           ##
###############################

#function run_auxgan_for_percentage() {  # SimLex && SimVerb
#    AUXGAN_DIR="../auxgan"
#    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
#    $PATH_TO_AUXGAN_PYTHON \
#    "$AUXGAN_DIR/code/adversarial.py" \
#    --seen_file $ORIGINAL_VECTORS"prefixed.txt" \
#    --adjusted_file "$CURR_DIR/$OUTDIR/ar$PERCENTAGEREP$ARVECTOR_POSTFIXFILENAME" \
#    --unseen_file $ORIGINAL_VECTORS"prefixed.txt" \
#    --out_dir $OUTDIR \
#    --n_epochs $N_EPOCHS \
#    --epoch_size $EPOCH_SIZE \
#    --map_optimizer $OPTIMIZER,lr=$LEARNING_RATE \
#    --dis_optimizer $OPTIMIZER,lr=$LEARNING_RATE \
#    --sim_optimizer $OPTIMIZER,lr=$LEARNING_RATE \
#    --in_simlex_file ../auxgan/evaluation/simlexorig999.txt \
#    --in_simverb_file ../auxgan/evaluation/simverb3500.txt \
#    --in_embedding_file $OUTDIR/gold_embs.txt
#    # --map_optimizer adam,lr=0.001 \
#    # --dis_optimizer adam,lr=0.001 \
#    # --sim_optimizer adam,lr=0.001
#    # echo "Evaluated simlexorig999 and simverb3500 for $OUTDIR" > "evaluated_simverb_$OUTDIR.txt"
#}

#function run_auxgan_for_percentage() {  # CARD
#    AUXGAN_DIR="../auxgan"
#    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
#    $PATH_TO_AUXGAN_PYTHON \
#    "$AUXGAN_DIR/code/adversarial.py" \
#    --seen_file $ORIGINAL_VECTORS"prefixed.txt" \
#    --adjusted_file "$CURR_DIR/$OUTDIR/ar$PERCENTAGEREP$ARVECTOR_POSTFIXFILENAME" \
#    --unseen_file $CARD_FT_VECS \
#    --out_dir $OUTDIR \
#    --n_epochs $N_EPOCHS \
#    --epoch_size $EPOCH_SIZE \
#    --map_optimizer $OPTIMIZER,lr=$LEARNING_RATE \
#    --dis_optimizer $OPTIMIZER,lr=$LEARNING_RATE \
#    --sim_optimizer $OPTIMIZER,lr=$LEARNING_RATE \
#    --in_card_file $CARD_FT_VECS \
#    --in_embedding_file $OUTDIR/gold_embs.txt
#    # --map_optimizer adam,lr=0.001 \
#    # --dis_optimizer adam,lr=0.001 \
#    # --sim_optimizer adam,lr=0.001
#    # echo "Evaluated simlexorig999 and simverb3500 for $OUTDIR" > "evaluated_simverb_$OUTDIR.txt"
#}

function run_auxgan_for_percentage() {  # SimLex && SimVerb && CARD
    AUXGAN_DIR="../auxgan"
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    $PATH_TO_AUXGAN_PYTHON \
    "$AUXGAN_DIR/code/adversarial.py" \
    --seen_file $ORIGINAL_VECTORS"prefixed.txt" \
    --adjusted_file "$CURR_DIR/$OUTDIR/ar$PERCENTAGEREP$ARVECTOR_POSTFIXFILENAME" \
    --unseen_file concatenated_fasttext_and_card_vectors.txt \
    --out_dir $OUTDIR \
    --n_epochs $N_EPOCHS \
    --epoch_size $EPOCH_SIZE \
    --map_optimizer $OPTIMIZER,lr=$LEARNING_RATE \
    --dis_optimizer $OPTIMIZER,lr=$LEARNING_RATE \
    --sim_optimizer $OPTIMIZER,lr=$LEARNING_RATE \
    --in_simlex_file ../auxgan/evaluation/simlexorig999.txt \
    --in_simverb_file ../auxgan/evaluation/simverb3500.txt \
    --in_card_file ../auxgan/evaluation/card660.tsv \
    --in_embedding_file $OUTDIR/gold_embs.txt
    # --map_optimizer adam,lr=0.001 \
    # --dis_optimizer adam,lr=0.001 \
    # --sim_optimizer adam,lr=0.001
    # echo "Evaluated simlexorig999 and simverb3500 for $OUTDIR" > "evaluated_simverb_$OUTDIR.txt"
}
