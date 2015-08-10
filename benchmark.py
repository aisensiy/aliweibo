#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LinearRegression
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s')

train_names = ['uid', 'mid', 'date', 'forward', 'comment', 'like', 'content']


def dev(pred, tt, base=5):
    pred = np.where(pred < 0, 0, pred)
    return np.abs(pred - tt) / (tt + base)


def score(f, c, l, cnt):
    cnt = np.where(cnt > 100, 100, cnt)
    prec = 1 - 0.5 * f - 0.25 * c - 0.25 * l
    return np.sum((cnt + 1) * np.where(prec > 0.8, 1.0, 0.0)) / np.sum(cnt + 1)


def basescore(data):
    m = 0.0
    return [m] * len(data)


def train_with_cv(df, pipeline, targets):
    skf = KFold(len(df), n_folds=3, shuffle=True, random_state=2014)
    scores = []
    # pipeline.set_params(**parameters)
    # print pipeline.get_params()

    for train_idx, test_idx in skf:
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        devs = []
        base_devs = []
        for target in targets:
            pipeline.fit(train['content'].values, train[target].values)
            predicted = pipeline.predict(test['content'].values)
            devs.append(dev(predicted.astype(int),
                            test[target].values,
                            target_bases[target]))
            base_devs.append(dev(basescore(test[target].values),
                                 test[target].values,
                                 target_bases[target]))
        s = score(devs[0], devs[1], devs[2], test['cnt'])
        base_s = score(base_devs[0], base_devs[1], base_devs[2], test['cnt'])
        scores.append(s)
        print 'score: %.6f' % s
        print 'base: %.6f' % base_s

    logging.info('Scores: %.6f' % np.mean(scores))


targets = ['forward', 'comment', 'like']
target_bases = {'forward': 5, 'comment': 3, 'like': 3}


if __name__ == '__main__':
    import sys
    pipeline = Pipeline([
        ('features', HashingVectorizer(ngram_range=(1, 4))),
        ('model', LinearRegression())
    ])
    logging.info('read file')
    df = pd.read_csv(sys.argv[1], sep='\t', quoting=3, names=train_names)
    df['cnt'] = df[targets].sum(axis=1)
    df = df[df['content'].notnull()]
    logging.info('training...')
    train_with_cv(df, pipeline, targets)
