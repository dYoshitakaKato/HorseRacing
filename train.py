import pandas as pd
import numpy as np
import datetime
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import requests
from bs4 import BeautifulSoup
import time
import re
from urllib.request import urlopen
import optuna.integration.lightgbm as lgb_o
from itertools import combinations, permutations
import update_data
from return_model import Return
import split_data

from horse_results import HorseResults
hr = HorseResults.read_pickle(['2021_horse_results.pickle'])

from peds import Peds
p = Peds.read_pickle(['2021_peds_results.pickle'])
p.encode()

from results import Results
r = Results.read_pickle(['2021_results.pickle'])
r.preprocessing()
r.merge_horse_results(hr, n_samples_list=[5, 9, 'all'])
r.merge_peds(p.peds_e)
r.process_categorical()

train, test = split_data.split_data(r.data_c)
train, valid = split_data.split_data(train)
#説明変数と目的変数に分ける。dateはこの後不要なので省く。単勝オッズも学習時には使わない。
X_train = train.drop(['rank', 'date', '単勝'], axis=1)
y_train = train['rank']
X_valid = valid.drop(['rank', 'date', '単勝'], axis=1)
y_valid = valid['rank']


#データセットを作成
lgb_train = lgb_o.Dataset(X_train.values, y_train.values)
lgb_valid = lgb_o.Dataset(X_valid.values, y_valid.values)

params = {
    'objective': 'binary', #今回は0or1の二値予測なのでbinaryを指定
    'random_state': 100
}

#チューニング実行
lgb_clf_o = lgb_o.train(params, lgb_train,
                        valid_sets=(lgb_train, lgb_valid),
                        verbose_eval=100,
                        early_stopping_rounds=10,
                        optuna_seed=100 #optunaのseed固定
                        )


# lgb_clf_o.params


train, test = split_data.split_data(r.data_c)

#説明変数と目的変数に分ける。dateはこの後不要なので省く。
X_train = train.drop(['rank', 'date', '単勝'], axis=1)
y_train = train['rank']
#2021/3/12追加： テストデータの単勝オッズはシミュレーション時に使用するので残しておく
X_test = test.drop(['rank', 'date'], axis=1)
y_test = test['rank']

lgb_clf = lgb.LGBMClassifier(**lgb_clf_o.params)
lgb_clf.fit(X_train.values, y_train.values)
