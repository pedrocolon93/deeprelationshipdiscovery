source run_oovtests_auxgan.sh

PERCENTAGE=0.1
CUDA_VISIBLE_DEVICES=1

generate_data_for_percentage && attractrepel_for_percentage && run_auxgan_for_percentage && echo "Ran $PERCENTAGE" > $PERCENTAGE.txt
