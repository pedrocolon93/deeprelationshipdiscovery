#!/usr/bin/env bash
#python3 convert_vec_to_formatted_text.py
if ! [ -f cc.en.300.vec.gz ]
then
 wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
cn5-vectors convert_fasttext -n 2000000 -l en cc.en.300.vec.gz unfitted.hd5
cn5-vectors retrofit -s 6 unfitted.hd5 ../../conceptnet5/data/assoc/reduced.csv fitted.hd5
cn5-vectors join_retrofit -s 6 fitted.hd5