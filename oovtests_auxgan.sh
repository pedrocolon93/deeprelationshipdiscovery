# Generate constraint data for percentages
#PATH_TO_AR_PYTHON=/Users/pedro/opt/anaconda3/envs/attractrepel/bin/python
# git clone https://github.com/nmrksic/attract-repel
# conda create -n attractrepel python=2.7 tensorflow=1.14; conda activate attractrepel
PATH_TO_AR_PYTHON=/home/pedro/anaconda3/envs/attractrepel/bin/python
#PATH_TO_RETROGAN_PYTHON=/Users/pedro/opt/anaconda3/envs/OOVconverter/bin/python
PATH_TO_RETROGAN_PYTHON=/home/pedro/anaconda3/envs/gputester2/bin/python
PATH_TO_AUXGAN_PYTHON=/home/yxin/anaconda3/envs/.retrogan_drd/bin/python
#PATH_TO_AR="/Users/pedro/Documents/Documents - Pedro’s MacBook Pro/git/attract-repel"
PATH_TO_AR="/media/pedro/Data/P-Data/attract-repel"
#ORIGINAL_VECTORS="/Users/pedro/PycharmProjects/OOVconverter/fasttext_model/cc.en.300.cut400k.vec"
ORIGINAL_VECTORS="/home/pedro/OOVconverter/fasttext_model/cc.en.300.vec"
ARVECTOR_POSTFIXFILENAME="cc.en.300.ar.vec"
CURR_DIR=$(pwd)
echo "Working in"
echo $CURR_DIR
PERCENTAGE=0.05
SEED=42
function generate_data_for_percentage() {
    PERCENTAGEREP=${PERCENTAGE/\./_}
    OUTDIR="oov_test_$PERCENTAGEREP/"
    echo "Working in $CURR_DIR/oov_test_$PERCENTAGEREP/"
    echo "Outputting AR TO:"
    echo "$CURR_DIR/oov_test_$PERCENTAGEREP/ar$PERCENTAGEREP$ARVECTOR_POSTFIXFILENAME"
    mkdir $OUTDIR
    CONSTRAINTS=synonyms.txt
    python oov_cutter_slsv.py --target_file testing/SimLex-999.txt --percentage_to_leave $PERCENTAGE --seed $SEED --output_dir "oov_test_$PERCENTAGEREP/"
    python oov_cutter_slsv.py --target_file testing/SimVerb-3500.txt --percentage_to_leave $PERCENTAGE --seed $SEED --output_dir "oov_test_$PERCENTAGEREP/"
    python oov_cutter_slsv_constraints.py --seen_words "oov_test_$PERCENTAGEREP/SimLex-999_cut_to_$PERCENTAGEREP.txt" --all_constraints $CONSTRAINTS --output_dir "oov_test_$PERCENTAGEREP/"
    python oov_cutter_slsv_constraints.py --seen_words "oov_test_$PERCENTAGEREP/SimVerb-3500_cut_to_$PERCENTAGEREP.txt" --all_constraints $CONSTRAINTS --output_dir "oov_test_$PERCENTAGEREP/"
    echo "Fusing both"
    cat oov_test_$PERCENTAGEREP/synonyms_reducedwith_SimLex-999_$PERCENTAGEREP.txt oov_test_$PERCENTAGEREP/synonyms_reducedwith_SimVerb-3500_$PERCENTAGEREP.txt > oov_test_$PERCENTAGEREP/synonyms_reducedwith_$PERCENTAGEREP.txt

#    python oov_cutter_slsv_constraints_removeoverlap.py  --simlexcut "oov_test_$PERCENTAGEREP/synonyms_reducedwith_SimLex-999_$PERCENTAGEREP.txt" --simverbcut "oov_test_$PERCENTAGEREP/synonyms_reducedwith_SimVerb-3500_$PERCENTAGEREP.txt" --outputfile "oov_test_$PERCENTAGEREP/synonyms_reducedwith_$PERCENTAGEREP.txt"
    CONSTRAINTS=antonyms.txt
    python oov_cutter_slsv.py --target_file testing/SimLex-999.txt --percentage_to_leave $PERCENTAGE --seed $SEED --output_dir "oov_test_$PERCENTAGEREP/"
    python oov_cutter_slsv.py --target_file testing/SimVerb-3500.txt --percentage_to_leave $PERCENTAGE --seed $SEED --output_dir "oov_test_$PERCENTAGEREP/"
    python oov_cutter_slsv_constraints.py --seen_words "oov_test_$PERCENTAGEREP/SimLex-999_cut_to_$PERCENTAGEREP.txt" --all_constraints $CONSTRAINTS --output_dir "oov_test_$PERCENTAGEREP/"
    python oov_cutter_slsv_constraints.py --seen_words "oov_test_$PERCENTAGEREP/SimVerb-3500_cut_to_$PERCENTAGEREP.txt" --all_constraints $CONSTRAINTS --output_dir "oov_test_$PERCENTAGEREP/"
    echo "Fusing both"
    cat oov_test_$PERCENTAGEREP/antonyms_reducedwith_SimLex-999_$PERCENTAGEREP.txt oov_test_$PERCENTAGEREP/antonyms_reducedwith_SimVerb-3500_$PERCENTAGEREP.txt > oov_test_$PERCENTAGEREP/antonyms_reducedwith_$PERCENTAGEREP.txt
#    python oov_cutter_slsv_constraints_removeoverlap.py  --simlexcut "oov_test_$PERCENTAGEREP/antonyms_reducedwith_SimLex-999_$PERCENTAGEREP.txt" --simverbcut "oov_test_$PERCENTAGEREP/antonyms_reducedwith_SimVerb-3500_$PERCENTAGEREP.txt" --outputfile "oov_test_$PERCENTAGEREP/antonyms_reducedwith_$PERCENTAGEREP.txt"
}

function attractrepel_for_percentage() {  # this function adds prefix "en_" to words
    PERCENTAGEREP=${PERCENTAGE/\./_}
    OUTDIR="oov_test_$PERCENTAGEREP/"
    python data_prep_retrogan.py --arconfigname "arconfig_$PERCENTAGE.config" --path_to_ar $PATH_TO_AR \
    --path_to_ar_python $PATH_TO_AR_PYTHON --synonyms "oov_test_$PERCENTAGEREP/synonyms_reducedwith_$PERCENTAGEREP.txt" \
    --antonyms "oov_test_$PERCENTAGEREP/antonyms_reducedwith_$PERCENTAGEREP.txt" --ccn $ORIGINAL_VECTORS --aroutput "$CURR_DIR/oov_test_$PERCENTAGEREP/ar$PERCENTAGEREP$ARVECTOR_POSTFIXFILENAME" \
    --output_dir "oov_test_$PERCENTAGEREP/"
}

function run_retro_gan_for_percentage() {  # Pedro's job
    EPOCHS=50
    PERCENTAGEREP=${PERCENTAGE/\./_}
    OUTDIR="oov_test_$PERCENTAGEREP/"
    echo "Runing with $PATH_TO_RETROGAN_PYTHON retrogan_trainer_attractrepel_working_pytorch.py --epochs $EPOCHS\
    \"$CURR_DIR/oov_test_$PERCENTAGEREP/original.hdf\" \"$CURR_DIR/oov_test_$PERCENTAGEREP/arvecs.hdf\" \
    \"retrogan_$PERCENTAGEREP\" \"oov_test_$PERCENTAGEREP/retrogan_$PERCENTAGEREP/\""
    CUDA_VISIBLE_DEVICES=1 $PATH_TO_RETROGAN_PYTHON retrogan_trainer_attractrepel_working_pytorch.py --epochs $EPOCHS\
    "$CURR_DIR/oov_test_$PERCENTAGEREP/original.hdf" "$CURR_DIR/oov_test_$PERCENTAGEREP/arvecs.hdf" \
    "retrogan_$PERCENTAGEREP" "oov_test_$PERCENTAGEREP/retrogan_$PERCENTAGEREP/"
}

function run_auxgan_for_percentage() {  # Yida's job
    EPOCHS=1
    PERCENTAGEREP=${PERCENTAGE/\./_}
    OUTDIR="oov_test_$PERCENTAGEREP/"
    AUXGAN_DIR="../../adversarial-postspec/"
    CUDA_VISIBLE_DEVICES=1 \
    $PATH_TO_AUXGAN_PYTHON \
    "$AUXGAN_DIR/code/adversarial.py" \
    --seen_file $ORIGINAL_VECTORS"prefixed.txt" \
    --adjusted_file "$CURR_DIR/oov_test_$PERCENTAGEREP/ar$PERCENTAGEREP$ARVECTOR_POSTFIXFILENAME" \
    --unseen_file $ORIGINAL_VECTORS"prefixed.txt" \
    --out_dir $OUTDIR \
    --n_epochs $EPOCHS \
    --map_optimizer adam,lr=0.001 \
    --dis_optimizer adam,lr=0.001 \
    --sim_optimizer adam,lr=0.001
}

PERCENTAGE=0.05
generate_data_for_percentage && attractrepel_for_percentage && run_auxgan_for_percentage && echo "Ran $PERCENTAGE">$PERCENTAGE.txt
#PERCENTAGE=0.1
#generate_data_for_percentage && attractrepel_for_percentage && run_auxgan_for_percentage && echo "Ran $PERCENTAGE">$PERCENTAGE.txt
#PERCENTAGE=0.25
#generate_data_for_percentage && attractrepel_for_percentage && run_auxgan_for_percentage && echo "Ran $PERCENTAGE">$PERCENTAGE.txt
#PERCENTAGE=0.5
#generate_data_for_percentage && attractrepel_for_percentage && run_auxgan_for_percentage && echo "Ran $PERCENTAGE">$PERCENTAGE.txt
#PERCENTAGE=0.75
#generate_data_for_percentage && attractrepel_for_percentage && run_auxgan_for_percentage && echo "Ran $PERCENTAGE">$PERCENTAGE.txt
#PERCENTAGE=1.0
#generate_data_for_percentage && attractrepel_for_percentage && run_auxgan_for_percentage && echo "Ran $PERCENTAGE">$PERCENTAGE.txt
