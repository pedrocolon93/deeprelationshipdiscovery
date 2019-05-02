from conceptnet5.nodes import standardized_concept_uri
from keract import get_activations, display_activations
import pandas as pd
import numpy as np
import gc
from keras.utils import plot_model

from deep_relationship_discovery import load_model_ours
drd_models_path = "trained_models/deepreldis/2019-04-2314:43:00.000000"
model_name = "UsedFor"
model = load_model_ours(save_folder=drd_models_path, model_name=model_name)[model_name]
plot_model(model)
model.summary()
retroembeddings = "trained_models/retroembeddings/2019-04-0813:03:02.430691/retroembeddings.h5"
retrofitted_embeddings = pd.read_hdf(retroembeddings, "mat")
w1 = np.array(retrofitted_embeddings.loc[standardized_concept_uri("en", "phone")]).reshape(1, 300)
w2 = np.array(retrofitted_embeddings.loc[standardized_concept_uri("en", "picture")]).reshape(1, 300)
w3 = np.array(retrofitted_embeddings.loc[standardized_concept_uri("en", "potato")]).reshape(1, 300)
del retrofitted_embeddings
gc.collect()
x={"retro_word_1": w1,"retro_word_2": w2}

print("For word 1")
layer_names = ["batch_normalization_1","dense_2","multiply_1","add_1"]
for name in layer_names:
    activations = get_activations(model, x, layer_name=name)
    display_activations(activations)
print('for word 2')
layer_names = ["batch_normalization_3","dense_4","multiply_2","add_2"]
for name in layer_names:
    activations = get_activations(model, x, layer_name=name)
    display_activations(activations)
