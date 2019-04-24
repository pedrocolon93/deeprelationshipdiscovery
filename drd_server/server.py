import json

from flask import Flask
import pandas as pd
from tensorflow import Session, Graph

import CNQuery
from flask import request

from deep_relationship_discovery import load_model_ours, normalize_outputs, relations
from tools import find_in_dataset, find_closest_in_dataset
from flask_cors import CORS
app = Flask(__name__)
global retro_e, drd_models
CORS(app)

app.route("/")
def hello():
    print(request.json)
    return "Hello World!"

@app.route("/get_neighbors",methods=['POST'])
def get_neighbors():
    global retro_e
    neighbors = []
    try:
        print("Checking neighbors")
        concept_to_explore = request.json["concept"]
        amount = request.json["amount"]
        print("for",concept_to_explore)
        concept_to_explore_vec = find_in_dataset([concept_to_explore],retro_e)
        print(concept_to_explore_vec)
        neighbors,_ = find_closest_in_dataset(concept_to_explore_vec,retro_e,n_top=amount,limit=100000)
        print(neighbors)
    except Exception as e:
        print(e)
    return json.dumps(neighbors)

@app.route("/get_relations",methods=['POST'])
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
    queryres = query.query(start,end,None)
    return json.dumps(queryres)
import tensorflow as tf

@app.route("/get_inferred_relations",methods=['POST'])
def get_inferred_relations():
    start = request.json["start"]
    end = request.json["end"]
    rel = request.json["rel"]
    start_vec,end_vec = find_in_dataset([start,end], retro_e)
    start_vec = start_vec.reshape(1,300)
    end_vec = end_vec.reshape(1,300)
    graph1 = Graph()
    with graph1.as_default():
        session1 = Session()
        with session1.as_default():
            drd_model = load_model_ours(save_folder=drd_models_path,model_name=rel)
            inferred_res = drd_model[rel].predict(x={"retro_word_1":start_vec,
                                             "retro_word_2":end_vec})
            print(inferred_res)
            try:
                norm_res = normalizers[rel].transform(inferred_res)

                norm_res = norm_res[0][0]
                return json.dumps([float(norm_res)])
            except:
                return json.dumps([float(inferred_res[0][0])])

@app.route("/get_available_models")
def get_available_models():
    return json.dumps([x.replace("/r/", "") for x in relations])


if __name__ == '__main__':
    retroembeddings_path = '../trained_models/retroembeddings/2019-04-0813:03:02.430691/retroembeddings.h5'
    retro_e = pd.read_hdf(retroembeddings_path,'mat')
    global drd_models_path
    drd_models_path = "../trained_models/deepreldis/2019-04-2314:43:00.000000"
    global normalizers
    normalizers = normalize_outputs(None,save_folder="../trained_models/deepreldis/2019-04-1614:43:00.000000")
    # drd_models = load_model_ours(save_folder=drd_models_path,model_name='all')
    app.run(
        host='0.0.0.0',
        port=4000
    )