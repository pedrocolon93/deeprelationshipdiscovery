source run_oovtests_auxgan.sh

PERCENTAGE=0.25
CUDA_VISIBLE_DEVICES=2

generate_data_for_percentage && attractrepel_for_percentage && run_auxgan_for_percentage && echo "Ran $PERCENTAGE" > $PERCENTAGE.txt
