#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
import numpy as np
from benchmark import train_names
from benchmark import targets
from benchmark import target_bases
from benchmark import dev
from benchmark import score


print 'reading file...'
df = pd.read_csv('data/weibo_train_data.txt',
                 sep='\t', quoting=3, names=train_names)
df['cnt'] = df[targets].sum(axis=1)
n = len(df)
print 'calculating...'
devs = [dev(np.zeros(n), df[target].values, target_bases[target])
        for target in targets]
print score(devs[0], devs[1], devs[2], df['cnt'])
