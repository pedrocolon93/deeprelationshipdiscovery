import random
import sys

from convert_to_rdf import parse_fun

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


def to_xml_rdf(triples_list, dest_folder="./"):
    dict_res = {"edges":[]}
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
    with open(dest_folder+"cn.xml","wb") as cnfile:
        data = ET.Element('benchmark')
        items = ET.SubElement(data, 'entries')
        for item in results:
            if len(item.getchildren())==0:
                continue
            else:
                for child in item.getchildren():
                    items.append(child)
        cnfile.write(ET.tostring(data))


def gen_text(xml_loc):
    pass


if __name__ == '__main__':
    address = input("Give me a web page that you would like to explore").strip()
    # Web page & Cleaned Page
    filename = scrape_site(address, overwrite=True)
    # Entities
    e_list = extract_entities(filename)
    # Discovery
    triples_list = generate_kg(e_list,
                               drd_models_path="./trained_models/deepreldis/2019-04-2314:43:00.000000",
                               target_file_loc='trained_models/retroembeddings/2019-10-22 11:57:48.878874/retroembeddings.h5',
                               trained_model_path="fasttext_model/trained_retrogan/2019-10-22 11:22:41.253615ftar/toretrogen.h5",
                               ft_dir="./fasttext_model/cc.en.300.bin"
                               )
    xml_loc = to_xml_rdf(triples_list, dest_folder="/Users/pedro/Documents/git/chimera/data/ConceptNet/raw/test/1/")
    gen_text(xml_loc)
    # Text
