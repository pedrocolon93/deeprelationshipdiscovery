source run_oovtests_auxgan.sh

PERCENTAGE=0.5
CUDA_VISIBLE_DEVICES=3

generate_data_for_percentage && attractrepel_for_percentage && run_auxgan_for_percentage && echo "Ran $PERCENTAGE" > $PERCENTAGE.txt
