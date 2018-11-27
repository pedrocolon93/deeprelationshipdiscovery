sudo apt install zip gzip -y
mkdir retrogan
cd retrogan
wget https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/numberbatch-en-17.06.txt.gz
gunzip numberbatch-en-17.06.txt.gz
mv numberbatch-en-17.06.txt numberbatch
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M-subword.vec.zip
unzip wiki-news-300d-1M-subword.vec.zip
