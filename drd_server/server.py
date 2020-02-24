import json
import numpy as np
from flask import Flask
from flask import request
from flask_cors import CORS
from tensorflow import Graph
from tensorflow_core.python import Session

import CNQuery
import deep_relationship_discovery
import tools
from knowledge_graph_generation import read_file, clean_file, generate_kg
from deep_relationship_discovery import get_embedding, load_model_ours, relations
from tools import find_in_dataset, find_closest_in_dataset


app = Flask(__name__)
app.config["drd_models_path"] = "../trained_models/deepreldis/2020-02-24 10:25:38.473155"
app.config["ft_model_path"] = "../fasttext_model/cc.en.300.bin"
app.config["retrogan_model_path"] = "../trained_models/retrogans/ft_full_alldata_feb11/"
CORS(app)

app.route("/")


def hello():
    print(request.json)
    return "Hello World!"


@app.route("/get_neighbors", methods=['POST'])
def get_neighbors():
    global retro_e
    neighbors = []
    try:
        print("Checking neighbors")
        concept_to_explore = request.json["concept"]
        amount = int(request.json["amount"])
        print("for", concept_to_explore)
        concept_to_explore_vec = find_in_dataset([concept_to_explore], retro_e)
        print(concept_to_explore_vec)
        print("Finding the neighbors")
        neighbors, _ = find_closest_in_dataset(concept_to_explore_vec, retro_e, n_top=int(amount))
        neighbors = list(neighbors)
        print("Done")
        print(neighbors)
    except Exception as e:
        print(e)
    return json.dumps(neighbors)


@app.route("/get_relations", methods=['POST'])
def get_relations():
    edgelist = []
    start = None
    end = None
    try:
        start = request.json["start"]
    except:
        pass
    try:
        end = request.json["end"]
    except:
        pass
    query = CNQuery.CNQuery()
    queryres = query.query(start, end, None)
    print(queryres)
    return json.dumps(queryres)


@app.route("/get_inferred_relations", methods=['POST'])
def get_inferred_relations():
    start = request.json["start"]
    end = request.json["end"]
    rel = request.json["rel"]
    start_vec = np.array(get_embedding(start)).reshape(1, tools.dimensionality)
    end_vec = np.array(get_embedding(end)).reshape(1, tools.dimensionality)
    # end_vec = end_vec.reshape(1, tools.dimensionality)
    graph1 = Graph()
    with graph1.as_default():
        session1 = Session()
        with session1.as_default():
            drd_model = load_model_ours(save_folder=app.config["drd_models_path"], model_name=rel)
            try:
                try:
                    inferred_res = drd_model[rel].predict(x={"retro_word_1": start_vec,
                                                             "retro_word_2": end_vec})
                except:
                    inferred_res = drd_model[rel].predict(x={"input_1": start_vec,
                                                             "input_2": end_vec})
            except Exception as e:
                print(e)
                return json.dumps([0.0])

            print(inferred_res)
            try:
                norm_res = inferred_res[0][0][0]
                # norm_res = normalizers[rel].transform(inferred_res)
                #
                # norm_res = norm_res[0][0]
                return json.dumps([float(norm_res)])
            except:
                return json.dumps([float(inferred_res[0][0][0])])


@app.route("/kg_from_webpage_test")
def test_kg_generation():
    filename = "../WebPages/travel_ny_bos.txt"
    contents = read_file(filename)
    clean_contents = clean_file(contents)
    res = generate_kg(clean_contents)
    return json.dumps(res)


@app.route("/get_available_models")
def get_available_models():
    return json.dumps([x.replace("/r/", "") for x in relations])


if __name__ == '__main__':
    tools.dimensionality = 300
    deep_relationship_discovery.rcgan_folder = app.config["retrogan_model_path"]
    deep_relationship_discovery.fasttext_folder = app.config["ft_model_path"]
    deep_relationship_discovery.rcgan = deep_relationship_discovery.load_things()
    app.run(
        host='0.0.0.0',
        port=4000
    )
