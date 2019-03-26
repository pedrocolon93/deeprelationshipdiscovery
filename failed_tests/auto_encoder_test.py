from keras import Sequential
from keras.layers import Dense, BatchNormalization

from tools import load_training_input_2, root_mean_squared_error

input_dim = 300


def add_dense(model, batch_norm=True,dense_size=512):
    model.add(Dense(dense_size,activation="relu"))
    if batch_norm:
        model.add(BatchNormalization())
    return model


def build_model():
    model = Sequential()
    model.add(Dense(300,input_shape=(input_dim,)))
    model.add(BatchNormalization())
    hidden_layers = 20
    for i in range(hidden_layers):
        model = add_dense(model)

    model.add(Dense(300,activation="tanh"))
    model.compile(optimizer="rmsprop",loss=root_mean_squared_error,metrics=["accuracy"])
    return model

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = load_training_input_2(limit=10000)
    model = build_model()
    model.fit(train_x,train_y,epochs=100, batch_size=128)
    print(model.evaluate(test_x,test_y))
