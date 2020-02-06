import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from src.constants import selected_fields, encoding_fields
from src.preprocessor import Preprocessor


def main():
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')

    X = train[selected_fields]

    X[encoding_fields].astype('category', copy=False)
    y = train['SalePrice']

    preprocessor = Preprocessor()
    preprocessor.train(X)
    X = preprocessor.transform(X)
    y = np.array(y)

    clf = RandomForestRegressor()
    score = cross_val_score(clf, X, y, cv=5)
    print(score)

    X_test = test[selected_fields]
    X_test[encoding_fields].astype('category', copy=False)
    X_test = preprocessor.transform(X_test)
    clf.fit(X, y)
    y_pred = clf.predict(X_test)

    result = pd.DataFrame({
        'SalePrice': y_pred
    }, index=test['Id'])
    result.to_csv('output.csv')


if __name__ == "__main__":
    main()
