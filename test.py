from v4 import TrAdaboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv(r'E:\untitled\create2\data\TR\CN0CN2CN3CN4_TR.csv').values
    # df = Imputer().fit_transform(df)
    # df = df.fillna(-9)
    X_train, y_train = df[:, :-1], df[:, -1]
    dq = pd.read_csv(r'E:\untitled\create2\data\TR\CN4+_TR.csv').values
    X_test, y_test = dq[:, :-1], dq[:, -1]
    # X, y = df[:, :-1], df[:, -1]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    tra = TrAdaboost()
    # tra.fit(X_train, X_test, y_train, y_test,50)
    tra.fit(X_train, X_test, y_train, y_test)
    y_pred = tra.predict(X_test)

    print(accuracy_score(y_test, y_pred))
