from keras import Sequential
from keras.layers import Dense

from gantests import load_training_input_2

input_dim = 300

def build_model():
    model = Sequential()
    model.add(Dense(300,input_shape=(input_dim,)))
    model.add(Dense(1024,activation="relu"))
    model.add(Dense(2048,activation="relu"))
    model.add(Dense(4096,activation="relu"))
    model.add(Dense(2048,activation="relu"))
    model.add(Dense(1024,activation="relu"))
    model.add(Dense(300,activation="tanh"))
    model.compile(optimizer="rmsprop",loss="mse",metrics=["accuracy"])
    return model

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = load_training_input_2(limit=10000)
    model = build_model()
    model.fit(train_x,train_y,epochs=1000, batch_size=64)
    print(model.evaluate(test_x,test_y,batch_size=32))
