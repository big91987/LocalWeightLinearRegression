import numpy as np
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool
import math

class LocalWeightLinearRegression(object):

    def __init__(self, k, n_jobs=1):
        self.k = k
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def predict(self, X):

        result = []
        n_batch = math.ceil(X.shape[0] / self.n_jobs)

        with Pool(processes=self.n_jobs) as pool:
            for i in range(n_batch):
                result += pool.map_async(self._predict_single_wrapper, [(self, x) for x in X[i * self.n_jobs: (i+1) * self.n_jobs]]).get()

        return np.array(result)

    def _predict_single(self, example):

        w = np.exp(-0.5 * np.sum(np.square(self.X - example), axis=1) / self.k)

        _model = LinearRegression()

        _model.fit(X=self.X, y=self.y, sample_weight=w)

        ret = _model.predict(X=example[np.newaxis, :])[0]
        return ret

    @staticmethod
    def _predict_single_wrapper(row):
        cls_instance, example = row
        return cls_instance._predict_single(example)


def test_lwlr():
    import pandas as pd
    data_dir = 'data'
    train_df = pd.read_csv('{}/height_train.csv'.format(data_dir))
    train_df['boy_dummy'] = train_df['boy_dummy'].astype(float)

    test_df = pd.read_csv('{}/height_test.csv'.format(data_dir))
    test_df['boy_dummy'] = test_df['boy_dummy'].astype(float)

    model = LocalWeightLinearRegression(k=3, n_jobs=4)
    model.fit(y=train_df.child_height, X=train_df.loc[:, ['father_height', 'mother_height', 'boy_dummy']].values)

    test_df['p1'] = model.predict(test_df.loc[:, ['father_height', 'mother_height', 'boy_dummy']].values)
    print(test_df)


if __name__ == '__main__':
    test_lwlr()
