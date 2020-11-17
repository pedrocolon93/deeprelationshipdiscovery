mkdir "results"
echo 'Experiment 1'
mkdir glove_disjoint_paperdata
cd glove_disjoint_paperdata
#/home/pedro/anaconda3/envs/gputester2/bin/python ../data_prep_retrogan.py ../../adversarial-postspec/post-specialized\ embeddings/distrib/glove_distrib.txt 750000 cleanedglove_vecs.txt ../clean_syn.txt ../clean_ant ../simlexsimverb.words glove_ar_disjoint.txt -dcn disjointglove -ccn completeglove.txt
/home/pedro/anaconda3/envs/gputester2/bin/python ../toauxgan.py completeglove.txt.hdf disjointglove.hdf
