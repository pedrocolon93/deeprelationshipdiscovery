# Directory: /home/yxin/research/projects/RetroGAN-DRD/retrogandeeprelationshipdiscovery

source evaluate_oovtests_auxgan.sh

OOVTEST_DIR=oovtest-1_0-sgd-lr-0_1
CUDA_VISIBLE_DEVICES=2

evaluate_simlex && evaluate_simverb
