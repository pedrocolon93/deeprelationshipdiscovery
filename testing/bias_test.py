import json
import os
import pandas as pd
import numpy as np
from keras.engine.saving import load_model
from keras.optimizers import Adam

import tools
from retrogan_trainer import ConstMultiplierLayer


def load_professions():
    professions_file = os.path.join('/Users/pedro/Documents/git/debiaswe/data', 'professions.json')
    with open(professions_file, 'r') as f:
        professions = json.load(f)
    print('Loaded professions\n' +
          'Format:\n' +
          'word,\n' +
          'definitional female -1.0 -> definitional male 1.0\n' +
          'stereotypical female -1.0 -> stereotypical male 1.0')
    return professions

if __name__ == '__main__':
    profs = load_professions()
    profession_words = [p[0] for p in profs]
    names = ["Emily", "Aisha", "Anne", "Keisha", "Jill", "Tamika", "Allison", "Lakisha", "Laurie", "Tanisha", "Sarah",
             "Latoya", "Meredith", "Kenya", "Carrie", "Latonya", "Kristen", "Ebony", "Todd", "Rasheed", "Neil",
             "Tremayne",
             "Geoffrey", "Kareem", "Brett", "Darnell", "Brendan", "Tyrone", "Greg", "Hakim", "Matthew", "Jamal", "Jay",
             "Leroy", "Brad", "Jermaine"]
    profession_words+=names
    # names = [tools.standardized_concept_uri("en",x).replace("/c/en/","")for x in names]
    # profession_words+=names
    # profession_words = names
    # Make sure they are in the vocab:
    print(profession_words)

    drd_models_path = "../trained_models/deepreldis/2019-04-2314:43:00.000000"
    target_file_loc = '/Users/pedro/PycharmProjects/OOVconverter/trained_models/retroembeddings/2019-05-15 11:47:52.802481/retroembeddings.h5'
    output_file_loc = '/Users/pedro/PycharmProjects/OOVconverter/trained_models/retroembeddings/2019-05-15 11:47:52.802481/retroembeddings_modified.h5'
    trained_model_path = "../trained_models/retrogans/2019-04-0721:33:44.223104/toretrogen.h5"
    # Load retrogan
    retrogan = load_model(trained_model_path,
                          custom_objects={"ConstMultiplierLayer": ConstMultiplierLayer},
                          compile=False)
    retrogan.compile(optimizer=Adam(), loss=['mae'])
    retrogan.load_weights(trained_model_path)
    # Load our vocabulary
    target_voc = pd.read_hdf(target_file_loc, 'mat')

    triples = []
    # beef up our vocab with missing entries
    clean_file_contents = profession_words
    in_dataset = tools.check_index_in_dataset(clean_file_contents, target_voc)
    for i, val in enumerate(in_dataset):
        if not val:
            missing_text = clean_file_contents[i]
            print(missing_text)
            # print("Missing:",missing_text)
            we = tools.generate_fastext_embedding(missing_text, ft_dir="../fasttext_model/cc.en.300.bin")
            # print("We:",we)
            if missing_text in names:
                print("Name")
                index = "/c/en/"+missing_text
            else:
                print("Not name")
                index = tools.standardized_concept_uri("en", missing_text)
            # print(index)
            rwe = tools.get_retrofitted_embedding(we, retrogan)
            # print("Retrofitted_embedding",rwe)
            df = pd.DataFrame(data=[rwe], index=[index])
            target_voc = target_voc.append(df)
            print(target_voc.shape)

    target_voc.to_hdf(output_file_loc,'mat')