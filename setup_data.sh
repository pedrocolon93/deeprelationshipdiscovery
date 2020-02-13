#echo 'Making disjoint glove paperdata'
#mkdir glove_disjoint_paperdata
#cd glove_disjoint_paperdata
##/home/pedro/anaconda3/envs/gputester2/bin/python ../data_prep_retrogan.py ../../adversarial-postspec/post-specialized\ embeddings/distrib/glove_distrib.txt 750000 cleanedglove_vecs.txt ../clean_syn.txt ../clean_ant ../simlexsimverb.words glove_ar_disjoint.txt -dcn disjointglove -ccn completeglove.txt
#/home/pedro/anaconda3/envs/gputester2/bin/python ../toauxgan.py completeglove.txt.hdf disjointglove.hdf
#
#cd ..
#echo 'Making disjoint fasttext paperdata'
#mkdir ft_disjoint_paperdata
#cd ft_disjoint_paperdata
##/home/pedro/anaconda3/envs/gputester2/bin/python ../data_prep_retrogan.py ../../adversarial-postspec/post-specialized\ embeddings/distrib/ft_distrib.txt 750000 cleanedft_vecs.txt ../clean_syn.txt ../clean_ant ../simlexsimverb.words ft_ar_disjoint.txt -dcn disjointfasttext -ccn completefastext.txt
#/home/pedro/anaconda3/envs/gputester2/bin/python ../toauxgan.py completefastext.txt.hdf disjointfasttext.hdf
#
#cd ..
#echo 'Making full glove paperdata'
#mkdir glove_full_paperdata
#cd glove_full_paperdata
##/home/pedro/anaconda3/envs/gputester2/bin/python ../data_prep_retrogan.py ../../adversarial-postspec/post-specialized\ embeddings/distrib/glove_distrib.txt 750000 cleanedglove_vecs.txt ../synonyms.txt ../antonyms.txt ../simlexsimverb.words glove_ar_full.txt -dcn fullglove -ccn completeglove.txt
#/home/pedro/anaconda3/envs/gputester2/bin/python ../toauxgan.py completeglove.txt.hdf fullglove.hdf
#
#cd ..
#echo 'Making full fasttext paperdata'
#mkdir ft_full_paperdata
#cd ft_full_paperdata
##/home/pedro/anaconda3/envs/gputester2/bin/python ../data_prep_retrogan.py ../../adversarial-postspec/post-specialized\ embeddings/distrib/ft_distrib.txt 750000 cleanedft_vecs.txt ../synonyms.txt ../antonyms.txt ../simlexsimverb.words ft_ar_full.txt -dcn fullfasttext -ccn completefastext.txt
#/home/pedro/anaconda3/envs/gputester2/bin/python ../toauxgan.py completefastext.txt.hdf fullfasttext.hdf

#cd ..

echo 'Making full fasttext all data '
mkdir ft_full_alldata
cd ft_full_alldata
/home/pedro/anaconda3/envs/gputester2/bin/python ../data_prep_retrogan.py ../fasttext_model/cc.en.300.vec 750000 cleanedft_vecs.txt ../synonyms.txt ../antonyms.txt ../simlexsimverb.words ft_ar_full.txt -dcn fullfasttext -ccn completefastext.txt
/home/pedro/anaconda3/envs/gputester2/bin/python ../toauxgan.py completefastext.txt.hdf fullfasttext.hdf

#cd ..
#echo 'Making disjoint fasttext all data'
#mkdir ft_disjoint_alldata
#cd ft_disjoint_alldata
##/home/pedro/anaconda3/envs/gputester2/bin/python ../data_prep_retrogan.py ../fasttext_model/cc.en.300.vec 750000 cleanedft_vecs.txt ../clean_syn.txt ../clean_ant ../simlexsimverb.words ft_ar_disjoint.txt -dcn disjointfasttext -ccn completefastext.txt
#/home/pedro/anaconda3/envs/gputester2/bin/python ../toauxgan.py completefastext.txt.hdf disjointfasttext.hdf

#cd ..
#echo 'Making full glove all data '
#mkdir glove_full_alldata
#cd glove_full_alldata
##/home/pedro/anaconda3/envs/gputester2/bin/python ../data_prep_retrogan.py ../../glove/glove.840B.300d.txt 750000 cleanedft_vecs.txt ../synonyms.txt ../antonyms.txt ../simlexsimverb.words glove_ar_full.txt -dcn fullglove -ccn completeglove.txt
#/home/pedro/anaconda3/envs/gputester2/bin/python ../toauxgan.py completeglove.txt.hdf fullglove.hdf
#
#cd ..
#echo 'Making disjoint glove all data'
#mkdir glove_disjoint_alldata
#cd glove_disjoint_alldata
##/home/pedro/anaconda3/envs/gputester2/bin/python ../data_prep_retrogan.py ../../glove/glove.840B.300d.txt 750000 cleanedft_vecs.txt ../clean_syn.txt ../clean_ant ../simlexsimverb.words glove_ar_disjoint.txt -dcn disjointglove -ccn completeglove.txt
#/home/pedro/anaconda3/envs/gputester2/bin/python ../toauxgan.py completeglove.txt.hdf disjointglove.hdf
#cd ..
