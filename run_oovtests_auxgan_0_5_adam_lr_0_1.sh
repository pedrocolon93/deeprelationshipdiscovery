source run_oovtests_auxgan.sh

PERCENTAGE=0.5
CUDA_VISIBLE_DEVICES=7

N_EPOCHS=10
EPOCH_SIZE=1000000
OPTIMIZER=adam
LEARNING_RATE=0.1

PERCENTAGEREP=${PERCENTAGE/\./_}
LEARNING_RATE_REP=${LEARNING_RATE/\./_}

OUTDIR="oovtest-$PERCENTAGEREP-$OPTIMIZER-lr-$LEARNING_RATE_REP/"

generate_data_for_percentage && attractrepel_for_percentage && run_auxgan_for_percentage && echo "Ran $PERCENTAGE" > $PERCENTAGE.txt
