import sys

import requests
import re
import os

from transformers import BertForNextSentencePrediction, BertConfig

sys.path.append('ReadabiliPy/')
from readabilipy import simple_json_from_html_string


def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

def scrape_site(address,filename=None, overwrite=False):
    response = requests.get(address)

    article = simple_json_from_html_string(response.text, content_digests=False, node_indexes=False,
                                           use_readability=False)
    if filename is None:
        filename = article["title"]
    if not overwrite and os.path.exists(filename):
        return
    else:
        with open(filename,'w') as f:
            for item in article["plain_text"]:
                f.write(item["text"]+"\n")
    return filename

if __name__ == '__main__':
    address = "https://www.tripsavvy.com/traveling-from-nyc-to-boston-1613034"
    scrape_site(address,overwrite=True,filename="nyc_bos")
