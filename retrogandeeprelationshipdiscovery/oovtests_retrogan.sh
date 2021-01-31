# Generate constraint data for percentages
#PATH_TO_AR_PYTHON=/home/pedro/anaconda3/envs/attractrepel/bin/python
#PATH_TO_RETROGAN_PYTHON=/Users/pedro/opt/anaconda3/envs/OOVconverter/bin/python
#PATH_TO_AR="/media/pedro/Data/P-Data/attract-repel"
#ORIGINAL_VECTORS="/home/pedro/OOVconverter/fasttext_model/cc.en.300.vec"
#PATH_TO_CONCAT_VECS="/Users/pedro/PycharmProjects/OOVconverter/fasttext_model/concatenated_fasttext_and_card_vectors.txt"
#ARVECTOR_POSTFIXFILENAME="cc.en.300.ar.vec"

PATH_TO_AR_PYTHON="/scratch/yxin/.yida_attractrepel/bin/python"  # peterchin5
#PATH_TO_AUXGAN_PYTHON="/scratch/yxin/.yida_auxgan/bin/python"  # peterchin5
PATH_TO_RETROGAN_PYTHON="/scratch/yxin/.yida_auxgan/bin/python"  # peterchin5
PATH_TO_AR="/scratch/yxin/RetroGAN-DRD/attract-repel"  # peterchin5
ORIGINAL_VECTORS="/scratch/yxin/RetroGAN-DRD/data/cc.en.300.vec"  # peterchin5
PATH_TO_CONCAT_VECS="/scratch/yxin/RetroGAN-DRD/retrogandeeprelationshipdiscovery/concatenated_fasttext_and_card_vectors.txt"  # peterchin5
CARD_FT_VECS="/scratch/yxin/RetroGAN-DRD/auxgan/evaluation/card_ft_vecs.txt"
ARVECTOR_POSTFIXFILENAME="cc.en.300.ar.vec"

CURR_DIR=$(pwd)
echo "Working in"
echo $CURR_DIR
PERCENTAGE=0.05
SEED=42

##################################
##                              ##
## Generate data for percentage ##
##                              ##
##################################

#function generate_data_for_percentage() {
#    local PERCENTAGEREP=${PERCENTAGE/\./_}
#    local OUTDIR="oov_test_$PERCENTAGEREP/"
#    echo "Working in $CURR_DIR/oov_test_$PERCENTAGEREP/"
#    echo "Outputting AR TO:"
#    echo "$CURR_DIR/oov_test_$PERCENTAGEREP/ar$PERCENTAGEREP$ARVECTOR_POSTFIXFILENAME"
#    mkdir $OUTDIR
#    python oov_cutter_slsv.py --target_file simlexsimverb_words.txt --percentage_to_leave $PERCENTAGE --seed $SEED --output_dir "oov_test_$PERCENTAGEREP/"
#
#    CONSTRAINTS=synonyms.txt
##    python oov_cutter_slsv.py --target_file testing/SimLex-999.txt --percentage_to_leave $PERCENTAGE --seed $SEED --output_dir "oov_test_$PERCENTAGEREP/"
##    python oov_cutter_slsv.py --target_file testing/SimVerb-3500.txt --percentage_to_leave $PERCENTAGE --seed $SEED --output_dir "oov_test_$PERCENTAGEREP/"
##    python oov_cutter_slsv_constraints.py --seen_words "oov_test_$PERCENTAGEREP/SimLex-999_cut_to_$PERCENTAGEREP.txt" --all_constraints $CONSTRAINTS --output_dir "oov_test_$PERCENTAGEREP/"
##    python oov_cutter_slsv_constraints.py --seen_words "oov_test_$PERCENTAGEREP/SimVerb-3500_cut_to_$PERCENTAGEREP.txt" --all_constraints $CONSTRAINTS --output_dir "oov_test_$PERCENTAGEREP/"
#    python oov_cutter_slsv_constraints.py --seen_words "oov_test_$PERCENTAGEREP/simlexsimverb_words_cut_to_$PERCENTAGEREP.txt" --all_constraints $CONSTRAINTS --output_dir "oov_test_$PERCENTAGEREP/"
#    echo "Fusing both"
##    cat oov_test_$PERCENTAGEREP/synonyms_reducedwith_SimLex-999_$PERCENTAGEREP.txt oov_test_$PERCENTAGEREP/synonyms_reducedwith_SimVerb-3500_$PERCENTAGEREP.txt > oov_test_$PERCENTAGEREP/synonyms_reducedwith_$PERCENTAGEREP.txt
#    cp oov_test_$PERCENTAGEREP/synonyms_reducedwith_simlexsimverb_$PERCENTAGEREP.txt oov_test_$PERCENTAGEREP/synonyms_reducedwith_$PERCENTAGEREP.txt
#
##    python oov_cutter_slsv_constraints_removeoverlap.py  --simlexcut "oov_test_$PERCENTAGEREP/synonyms_reducedwith_SimLex-999_$PERCENTAGEREP.txt" --simverbcut "oov_test_$PERCENTAGEREP/synonyms_reducedwith_SimVerb-3500_$PERCENTAGEREP.txt" --outputfile "oov_test_$PERCENTAGEREP/synonyms_reducedwith_$PERCENTAGEREP.txt"
#    CONSTRAINTS=antonyms.txt
##    python oov_cutter_slsv.py --target_file testing/SimLex-999.txt --percentage_to_leave $PERCENTAGE --seed $SEED --output_dir "oov_test_$PERCENTAGEREP/"
##    python oov_cutter_slsv.py --target_file testing/SimVerb-3500.txt --percentage_to_leave $PERCENTAGE --seed $SEED --output_dir "oov_test_$PERCENTAGEREP/"
##    python oov_cutter_slsv_constraints.py --seen_words "oov_test_$PERCENTAGEREP/SimLex-999_cut_to_$PERCENTAGEREP.txt" --all_constraints $CONSTRAINTS --output_dir "oov_test_$PERCENTAGEREP/"
##    python oov_cutter_slsv_constraints.py --seen_words "oov_test_$PERCENTAGEREP/SimVerb-3500_cut_to_$PERCENTAGEREP.txt" --all_constraints $CONSTRAINTS --output_dir "oov_test_$PERCENTAGEREP/"
#    python oov_cutter_slsv_constraints.py --seen_words "oov_test_$PERCENTAGEREP/simlexsimverb_words_cut_to_$PERCENTAGEREP.txt" --all_constraints $CONSTRAINTS --output_dir "oov_test_$PERCENTAGEREP/"
#
#    echo "Fusing both"
#    cp oov_test_$PERCENTAGEREP/antonyms_reducedwith_simlexsimverb_$PERCENTAGEREP.txt oov_test_$PERCENTAGEREP/antonyms_reducedwith_$PERCENTAGEREP.txt
#
##    cat oov_test_$PERCENTAGEREP/antonyms_reducedwith_SimLex-999_$PERCENTAGEREP.txt oov_test_$PERCENTAGEREP/antonyms_reducedwith_SimVerb-3500_$PERCENTAGEREP.txt > oov_test_$PERCENTAGEREP/antonyms_reducedwith_$PERCENTAGEREP.txt
##    python oov_cutter_slsv_constraints_removeoverlap.py  --simlexcut "oov_test_$PERCENTAGEREP/antonyms_reducedwith_SimLex-999_$PERCENTAGEREP.txt" --simverbcut "oov_test_$PERCENTAGEREP/antonyms_reducedwith_SimVerb-3500_$PERCENTAGEREP.txt" --outputfile "oov_test_$PERCENTAGEREP/antonyms_reducedwith_$PERCENTAGEREP.txt"
#}

function generate_data_for_percentage() {
    local PERCENTAGEREP=${PERCENTAGE/\./_}
    local OUTDIR="oov_test_$PERCENTAGEREP/"
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

##################################
##                              ##
## attract-repel for percentage ##
##                              ##
##################################

function attractrepel_for_percentage() {
    local PERCENTAGEREP=${PERCENTAGE/\./_}
    local OUTDIR="oov_test_$PERCENTAGEREP/"
    local ADAM_AUXGAN_DIR="oovtest-$PERCENTAGEREP-adam-lr-0_1"
    python data_prep_retrogan.py --arconfigname "arconfig_$PERCENTAGE.config" --path_to_ar $PATH_TO_AR \
    --path_to_ar_python $PATH_TO_AR_PYTHON --synonyms "$OUTDIR/synonyms_reducedwith_$PERCENTAGEREP.txt" \
    --antonyms "$OUTDIR/antonyms_reducedwith_$PERCENTAGEREP.txt" --ccn $ORIGINAL_VECTORS --aroutput "$CURR_DIR/$OUTDIR/ar$PERCENTAGEREP$ARVECTOR_POSTFIXFILENAME" \
    --output_dir "$OUTDIR/" --skip_ar --skip_prefix \
    --origvectors $PATH_TO_CONCAT_VECS \
    --arvectors "$CURR_DIR/$ADAM_AUXGAN_DIR/ar$PERCENTAGEREP$ARVECTOR_POSTFIXFILENAME"
}

##################
##              ##
## run retrogan ##
##              ##
##################

function run_retro_gan_for_percentage() {
    ITERS=100000
    EPOCHS=250
    local PERCENTAGEREP=${PERCENTAGE/\./_}
    local OUTDIR="oov_test_$PERCENTAGEREP/"
    # echo "Runing with $PATH_TO_RETROGAN_PYTHON retrogan_trainer_attractrepel_working_pytorch.py --iters $ITERS\
    # \"$CURR_DIR/oov_test_$PERCENTAGEREP/original.hdf\" \"$CURR_DIR/oov_test_$PERCENTAGEREP/arvecs.hdf\" \
    # \"retrogan_$PERCENTAGEREP\" \"oov_test_$PERCENTAGEREP/retrogan_$PERCENTAGEREP/\""
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $PATH_TO_RETROGAN_PYTHON retrogan_trainer_attractrepel_working_pytorch.py --iters $ITERS\
    "$CURR_DIR/$OUTDIR/original.hdf" "$CURR_DIR/$OUTDIR/arvecs.hdf" \
    "retrogan_$PERCENTAGEREP" "$OUTDIR/retrogan_$PERCENTAGEREP/" --fp16
}
function run05() {
    local PERCENTAGE=0.05
    local VISDEV=5
    local CUDA_VISIBLE_DEVICES=$VISDEV
    generate_data_for_percentage &&
    attractrepel_for_percentage &&
    run_retro_gan_for_percentage && echo "Ran $PERCENTAGE">$PERCENTAGE.txt
}
function run10() {
    local PERCENTAGE=0.1
    local VISDEV=6
    local CUDA_VISIBLE_DEVICES=$VISDEV
    generate_data_for_percentage &&
    attractrepel_for_percentage &&
    run_retro_gan_for_percentage && echo "Ran $PERCENTAGE">$PERCENTAGE.txt
}
function run25() {
    local PERCENTAGE=0.25
    local VISDEV=7
    local CUDA_VISIBLE_DEVICES=$VISDEV
    generate_data_for_percentage &&
    attractrepel_for_percentage &&
    run_retro_gan_for_percentage && echo "Ran $PERCENTAGE">$PERCENTAGE.txt
}
function run50() {
    local PERCENTAGE=0.5
    local VISDEV=7
    local CUDA_VISIBLE_DEVICES=$VISDEV
    generate_data_for_percentage &&
    attractrepel_for_percentage &&
    run_retro_gan_for_percentage && echo "Ran $PERCENTAGE">$PERCENTAGE.txt
}
function run75() {
    local PERCENTAGE=0.75
    local VISDEV=6
    local CUDA_VISIBLE_DEVICES=$VISDEV
    generate_data_for_percentage &&
    attractrepel_for_percentage &&
    run_retro_gan_for_percentage && echo "Ran $PERCENTAGE">$PERCENTAGE.txt
}
function run100() {
    local PERCENTAGE=1.0
    local VISDEV=5
    local CUDA_VISIBLE_DEVICES=$VISDEV
    generate_data_for_percentage &&
    attractrepel_for_percentage &&
    run_retro_gan_for_percentage && echo "Ran $PERCENTAGE">$PERCENTAGE.txt
}
run05 &
#run10 &
#run25 &
#run50 &
#run75 &
#run100
