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
from tqdm.notebook import tqdm
import re
from urllib.request import urlopen
import optuna.integration.lightgbm as lgb_o
from itertools import combinations, permutations

#払い戻し表データを処理するクラス
class Return:
    @staticmethod
    def scrape(race_id_list):
        """
        払い戻し表データをスクレイピングする関数

        Parameters:
        ----------
        race_id_list : list
            レースIDのリスト

        Returns:
        ----------
        return_tables_df : pandas.DataFrame
            全払い戻し表データをまとめてDataFrame型にしたもの
        """

        return_tables = {}
        for race_id in tqdm(race_id_list):
            time.sleep(0.5)
            try:
                url = "https://db.netkeiba.com/race/" + race_id

                #普通にスクレイピングすると複勝やワイドなどが区切られないで繋がってしまう。
                #そのため、改行コードを文字列brに変換して後でsplitする
                f = urlopen(url)
                html = f.read()
                html = html.replace(b'<br />', b'br')
                dfs = pd.read_html(html)

                #dfsの1番目に単勝〜馬連、2番目にワイド〜三連単がある
                df = pd.concat([dfs[1], dfs[2]])

                df.index = [race_id] * len(df)
                return_tables[race_id] = df
            except IndexError:
                continue
            except Exception as e:
                print(e)
                break
            except:
                break

        #pd.DataFrame型にして一つのデータにまとめる
        return_tables_df = pd.concat([return_tables[key] for key in return_tables])
        return return_tables_df