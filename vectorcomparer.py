import numpy as np
def find_vectors(concept1,concept2):
    vector1 = None
    vector2 = None
    with open('retrogan/numberbatch','r') as f:
        for line in f:
            split = line.split(" ")
            if split[0].strip() == concept1:
                vector1 = np.array([float(x) for x in split[1:]])
            if split[0].strip() == concept2:
                vector2 = np.array([float(x) for x in split[1:]])
            if vector1 is not None and vector2 is not None:
                break
    return vector1,vector2