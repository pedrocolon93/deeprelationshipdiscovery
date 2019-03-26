import os
import pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

from tools import load_training_input_2
from sklearn.metrics import mean_absolute_error

if __name__ == '__main__':
    X_train = Y_train = X_test = Y_test = None

    regen = False
    normalize = False
    seed = 32
    file = "data.pickle"
    if not os.path.exists(file) or regen:
        X_train, Y_train, X_test, Y_test = load_training_input_2(normalize=normalize, seed=seed, test_split=0.1)
        pickle.dump((X_train, Y_train, X_test, Y_test), open(file, "wb"))
    else:
        X_train, Y_train, X_test, Y_test = pickle.load(open("data.pickle", 'rb'))

    # gpr = GaussianProcessRegressor()
    gpr = MultiOutputRegressor(SVR(kernel='rbf',verbose=True,gamma="scale"), n_jobs=8)

    gpr.fit(X_train,Y_train)
    ypred = gpr.predict(X_test)
    print(mean_absolute_error(Y_test,ypred))