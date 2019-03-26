import numpy
import pandas
from keras import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold, cross_val_score

df = pandas.read_csv("./dataset/housing.data",delim_whitespace=True,header=None)
dataset = df.values

X =dataset[:,0:13]
Y =dataset[:,13]

def baseline_model():
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

seed = 7
numpy.random.seed(seed)
estimator = KerasRegressor(build_fn=baseline_model,epochs=100,batch_size=5,verbose=0)

kfold = KFold(n_splits=10,random_state=seed)
results = cross_val_score(estimator,X,Y,cv=kfold)
print(results.mean(),results.std())