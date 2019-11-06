import random
import shutil
import sys
import pandas as pd
from tqdm import tqdm

from convert_to_rdf import parse_fun
import subprocess

# sys.path.append("/Users/pedro/Documents/git/chimera/")
# from data.ConceptNet.reader import ConceptNetDataReader
# from data.reader import DataReader, DataSetType
# from planner.neural_planner import NeuralPlanner
# from planner.planner import Planner
# from process.evaluation import EvaluationPipeline
# from process.pre_process import TrainingPreProcessPipeline, TestingPreProcessPipeline
# from process.reg import REGPipeline
# from process.train_model import TrainModelPipeline
# from process.train_planner import TrainPlannerPipeline
# from process.translate import TranslatePipeline
# from reg.base import REG
# from reg.bert import BertREG
# from utils.pipeline import Pipeline

import xml.etree.ElementTree as ET

from extract_entities import extract_entities
from knowledge_graph_generation import generate_kg
from scrape_site import scrape_site


def to_xml_rdf(triples_list, dest_folder="./", filename="def.xml"):
    dict_res = {"edges": []}
    for triple in triples_list:
        # c c s r
        json_dict = {}
        start = {"term": triple[0]}
        end = {"term": triple[1]}
        rel = {"@id": triple[3]}
        json_dict["start"] = start
        json_dict["end"] = end
        json_dict["rel"] = rel
        json_dict["surfaceText"] = ""
        dict_res["edges"].append(json_dict)
    res = parse_fun(dict_res)
    results = [res]
    with open(dest_folder + filename, "wb") as cnfile:
        data = ET.Element('benchmark')
        items = ET.SubElement(data, 'entries')
        for item in tqdm(results):
            if len(item.getchildren()) == 0:
                continue
            else:
                for child in tqdm(item.getchildren()):
                    items.append(child)
        cnfile.write(ET.tostring(data))
    return dest_folder + filename




def gen_text():
    chimera_env_loc = "/Users/pedro/anaconda3/envs/chimera/bin/python"
    chimera_location = "/Users/pedro/Documents/git/chimera/"

    # # Move the test outputs
    # try:
    #     print("Moving file")
    #     shutil.move(xml_loc, chimera_location+"data/ConceptNet/raw/test/1/")
    # except Exception as e:
    #     print(e)
    # Remove cache
    try:
        print("Removing cache")
        shutil.rmtree(chimera_location+"cache/ConceptNet/test-corpus")
    except Exception as e:
        print(e)
    print("Opening subprocess")
    MyOut = subprocess.Popen([chimera_env_loc, chimera_location+"main.py"],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
    print("Done.Communicating")
    while True:
        output = MyOut.stdout.readline()
        if output == '' and MyOut.poll() is not None:
            break
        if output:
            print( output.strip())
    rc = MyOut.poll()
    # stdout, stderr = MyOut.communicate()
    print("Done")
    # print(stdout)
    # print(stderr)


def clean_trips(triples_list):
    averages_dict = {}
    rel_count = {}
    rel_tot = {}
    print("Filtering list")
    for triple in triples_list:
        if triple[3] not in averages_dict:
            averages_dict[triple[3]] = 0
        if triple[3] not in rel_count:
            rel_count[triple[3]] = 0
        if triple[3] not in rel_tot:
            rel_tot[triple[3]] = 0
        rel = triple[3]
        rel_count[rel] += 1
        rel_tot[rel] += triple[2]
    for rel in averages_dict:
        averages_dict[rel] = (rel_tot[rel] * 1.0) / rel_count[rel]

    def filter_fun(x):
        scale = 1.0
        return x[2] > scale * averages_dict[x[3]]

    cleaned_list = [x for x in filter(filter_fun, triples_list)]
    print(len(cleaned_list))
    for i in range(len(cleaned_list)):
        triple = cleaned_list[i]
        triple[3] = "/r/" + triple[3]
    return cleaned_list


if __name__ == '__main__':
    # address = input("Give me a web page that you would like to explore").strip()
    # Web page & Cleaned Page
    address = "https://www.tripsavvy.com/traveling-from-nyc-to-boston-1613034"
    filename = scrape_site(address, overwrite=True)
    # Entities
    e_list = extract_entities(filename)
    # Discovery
    triples_list = generate_kg(e_list,
                               limit=None,
                               drd_models_path="trained_models/deepreldis/2019-10-22 21:43:59.788678",
                               target_file_loc='trained_models/retroembeddings/2019-10-22 11:57:48.878874/retroembeddings.h5',
                               trained_model_path="fasttext_model/trained_retrogan/2019-10-22 11:22:41.253615ftar/toretrogen.h5",
                               ft_dir="./fasttext_model/cc.en.300.bin"
                               )
    cleaned_triples = clean_trips(triples_list)
    # pd.DataFrame(data=cleaned_triples,index=[x for x in range(len(cleaned_triples))]).to_hdf("checkpoint","mat")
    # triples_list = pd.read_hdf("checkpoint","mat")
    to_xml_rdf(triples_list.values, dest_folder="/Users/pedro/Documents/git/chimera/data/ConceptNet/raw/test/1/",
                         filename="myfile")
    gen_text()
    # Text
