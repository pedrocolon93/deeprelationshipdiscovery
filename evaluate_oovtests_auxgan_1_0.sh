# Directory: /home/yxin/research/projects/RetroGAN-DRD/retrogandeeprelationshipdiscovery

source evaluate_oovtests_auxgan.sh

OOVTEST_DIR=oov_test_1_0
CUDA_VISIBLE_DEVICES=1

evaluate_simlex && evaluate_simverb
