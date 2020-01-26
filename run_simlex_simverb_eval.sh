#!/usr/bin/env bash
cd /media/pedro/ssd_ext/adversarial-postspec/evaluation/
pwd
#conda activate attract_repel
/home/pedro/anaconda3/envs/adversarialpostspec/bin/python simlex_evaluator.py simlexorig999.txt /media/pedro/ssd_ext/attract-repel/results/glove_ar_disjoint.txt
/home/pedro/anaconda3/envs/adversarialpostspec/bin/python simlex_evaluator.py simlexorig999.txt /media/pedro/ssd_ext/attract-repel/results/glove_ar_disjoint.txt
#source deactivate