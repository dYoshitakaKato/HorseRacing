import pandas as pd
import time
#from tqdm import tqdm_notebook as tqdm
#現在ではインポートの仕方が下のように変わっています
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor

#レースIDのリストを作る
from results import Results
from return_model import Return
from horse_results import HorseResults
from peds import Peds

results = []
for year in range(2021, 1980, -1):
    race_id_list = []
    for place in range(1, 11, 1):
        for kai in range(1, 6, 1):
            for day in range(1, 13, 1):
                for r in range(1, 13, 1):
                    race_id = str(year) + str(place).zfill(2) + str(kai).zfill(2) +\
                    str(day).zfill(2) + str(r).zfill(2)
                    race_id_list.append(race_id)
    executor = ThreadPoolExecutor(max_workers=2)
    futures = []
    future = executor.submit(Results.scrape, race_id_list)
    futures.append(future)

    future = executor.submit(Return.scrape, race_id_list)
    futures.append(future)

    saves = [str(year) + '_results.pickle',  str(year) + '_returns_results.pickle']
    for i in range(0, 2, 1):
        r = futures[i].result()
        r.to_pickle(saves[i])
        pass
    results.append(futures[0].result())
    executor.shutdown()


return_tables_df = None
for res in results:
    return_tables_df = pd.concat([results[key] for key in return_tables])
horse_id_list = results[0]['horse_id'].unique()

executor = ThreadPoolExecutor(max_workers=2)
futures = []
future = executor.submit(HorseResults.scrape, horse_id_list)
futures.append(future)

future = executor.submit(Peds.scrape, horse_id_list)
futures.append(future)

saves = [str(year) + '_horse_results.pickle', str(year) + '_peds_results.pickle']
for i in range(0, 2, 1):
    r = futures[i].result()
    r.to_pickle(saves[i])
    results.append(r)
    pass
executor.shutdown()
