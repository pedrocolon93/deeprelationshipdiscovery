source run_oovtests_auxgan.sh

PERCENTAGE=0.75
CUDA_VISIBLE_DEVICES=0

OPTIMIZER=sgd
LEARNING_RATE=0.1

PERCENTAGEREP=${PERCENTAGE/\./_}
LEARNING_RATE_REP=${LEARNING_RATE/\./_}

OUTDIR="oovtest-$PERCENTAGEREP-$OPTIMIZER-lr-$LEARNING_RATE_REP/"

generate_data_for_percentage && attractrepel_for_percentage && run_auxgan_for_percentage && echo "Ran $PERCENTAGE" > $PERCENTAGE.txt
