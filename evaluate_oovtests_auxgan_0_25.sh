# Directory: /home/yxin/research/projects/RetroGAN-DRD/retrogandeeprelationshipdiscovery

source evaluate_oovtests_auxgan.sh

OOVTEST_DIR=oov_test_0_25
CUDA_VISIBLE_DEVICES=2

evaluate_simlex && evaluate_simverb
