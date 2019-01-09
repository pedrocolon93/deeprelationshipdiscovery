from keras import Sequential
from keras.layers import Dense, Activation, BatchNormalization
from keras.optimizers import Adam
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

from tools import load_training_input_2

X_train, Y_train, X_test, Y_test = load_training_input_2(100000)

# def make_model ():
#     model = Sequential()
#     vector_dim = 300
#     # model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
#     model.add(Dense(512, activation='relu', input_dim=vector_dim))
#     model.add(Dense(256, activation='relu'))
#     model.add(BatchNormalization(momentum=0.8))
#     model.add(Dense(128, activation='relu'))
#     model.add(BatchNormalization(momentum=0.8))
#     model.add(Dense(vector_dim))
#     optimizer = Adam()
#     model.compile(optimizer=optimizer,loss="mean_squared_error",metrics=["accuracy"])
#     return model
# estimator = KerasRegressor(build_fn=make_model, epochs=200, batch_size=64, verbose=1)

# estimator = GaussianProcessRegressor()
estimator = MultiOutputRegressor(SVR(),8)
estimator.fit(X_train,Y_train)
Y_predict = estimator.predict(X_test)
print(r2_score(Y_test,Y_predict))
