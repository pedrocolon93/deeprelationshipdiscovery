import multiprocessing
from multiprocessing import Queue
from multiprocessing.pool import Pool

from keras import Input, Model
from keras.layers import Dense, Concatenate
from keras.optimizers import Adam
from keras.utils import plot_model

from gan_tester import attention
import csv

def create_model():
    # Input needs to be 2 word vectors
    wv1 = Input(shape=(300,), name="Retro_word_vector_1")
    wv2 = Input(shape=(300,), name="Retro_word_vector_2")
    expansion_size = 512
    # Expand and contract the 2 word vectors
    wv1_expansion_1 = Dense(expansion_size)(wv1)
    wv1_expansion_2 = Dense(int(expansion_size / 2))(wv1_expansion_1)
    wv1_expansion_3 = Dense(int(expansion_size / 4))(wv1_expansion_2)
    wv2_expansion_1 = Dense(expansion_size)(wv2)
    wv2_expansion_2 = Dense(int(expansion_size / 2))(wv2_expansion_1)
    wv2_expansion_3 = Dense(int(expansion_size / 4))(wv2_expansion_2)
    # Concatenate both expansions
    merge1 = Concatenate()([wv1_expansion_3, wv2_expansion_3])
    merge_expand = Dense(expansion_size)(merge1)
    # Add atention layer
    merge_attention = attention(merge_expand)
    attention_expand = Dense(expansion_size)(merge_attention)
    semi_final_layer = Dense(expansion_size)(attention_expand)
    # Output layer
    amount_of_relations = 40
    # One big layer
    # final = Dense(amount_of_relations)(semi_final_layer)
    # Many tasks
    tl_neurons = 20
    final = []
    for i in range(amount_of_relations):
        task_layer = Dense(tl_neurons)(semi_final_layer)
        final.append(Dense(1, name=str(i))(task_layer))
    losses = []
    for i in range(amount_of_relations):
        loss = "mean_squared_error"
        losses.append(loss)
    drd = Model([wv1, wv2], final)
    optimizer = Adam()
    drd.compile(optimizer=optimizer, loss=losses)
    drd.summary()
    plot_model(drd)
    return drd

q = Queue()
def initializer():
    global q
    pass
def trio(relation,start,end):
    tuple = (relation,start,end)
    q.put(tuple)

def get_assertions():
    global q
    pool = Pool(multiprocessing.cpu_count(),initializer=initializer)

    with open("retrogan/conceptnet-assertions-5.6.0.csv") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            pool.apply_async(trio,args=(row[1], row[2], row[3]))

    pool.join()
    pool.close()
    print(len(q))
    return list(q)


if __name__ == '__main__':
    # model = create_model()
    filtered_assertions = get_assertions()

    # Output needs to be the relationship weights