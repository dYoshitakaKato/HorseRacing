import pandas as pd
import time
#from tqdm import tqdm_notebook as tqdm
#現在ではインポートの仕方が下のように変わっています
from tqdm.notebook import tqdm

def scrape_race_results(race_id_list, pre_race_results={}):
    #race_results = pre_race_results
    race_results = pre_race_results.copy() #正しくはこちら。注意点で解説。
    for race_id in tqdm(race_id_list):
        if race_id in race_results.keys():
            continue
        # print("race id is " + str(race_id))
        time.sleep(1)
        try:
            url = "https://db.netkeiba.com/race/" + race_id
            race_results[race_id] = pd.read_html(url)[0]
        except IndexError:
            continue
        #この部分は動画中に無いですが、捕捉できるエラーは拾った方が、エラーが出たときに分かりやすいです
        except Exception as e:
            print(e)
            break
        except:
            break
    return race_results

#レースIDのリストを作る
race_id_list = []
for year in range(1980, 2022, 1):
    for place in range(1, 11, 1):
        for kai in range(1, 6, 1):
            for day in range(1, 13, 1):
                for r in range(1, 13, 1):
                    race_id = str(year) + str(place).zfill(2) + str(kai).zfill(2) +\
                    str(day).zfill(2) + str(r).zfill(2)
                    race_id_list.append(race_id)

#スクレイピングしてデータを保存
test3 = scrape_race_results(race_id_list)
for key in test3: #.keys()は無くても大丈夫です
    test3[key].index = [key] * len(test3[key])
results = pd.concat([test3[key] for key in test3], sort=False)
results.to_pickle('results.pickle')