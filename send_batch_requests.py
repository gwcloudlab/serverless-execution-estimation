
import math
import sys
from threading import Thread
import os
import random
import pandas as pd
import requests

def worker(image_path, estimated_excution_time, sledge_url, rps, ts, i):
    request = 'hey -disable-compression -disable-keepalive -disable-redirects -H "Expected_Cost: {}" -c 4 -o csv -t 0 -q {} -z {}s -m POST -D "{}" "{}" >> result{}.csv'.format(estimated_excution_time,rps,ts,image_path,sledge_url, i)
    os.system(request)

# Credit: https://cmdlinetips.com/2022/07/randomly-sample-rows-from-a-big-csv-file/ 
def sample_n_from_csv(filename:str, n:int=100, total_rows:int=None) -> pd.DataFrame:
    if total_rows==None:
        with open(filename,"r") as fh:
            total_rows = sum(1 for row in fh)
    if(n>total_rows):
        print("Error: n > total_rows", file=sys.stderr) 
    skip_rows =  random.sample(range(1,total_rows+1), total_rows-n)
    return pd.read_csv(filename, skiprows=skip_rows)


if __name__=="__main__":

    requests_per_second = int(sys.argv[1])
    total_seconds = int(sys.argv[2])
    sledge_url = sys.argv[3]
    

    predictions_index = "estimated_time" + sledge_url.split("/")[-2][len("resize"):]

    predictions = sample_n_from_csv(filename="predictions.csv", n=30)
    # url,filename,estimated_time_small,estimated_time_medium,estimated_time_large
    path =  os.path.join(os.path.dirname(os.path.realpath(__file__)), 'images')
    if not os.path.exists(path):
      os.mkdir(path)
    
    for i in range(0, len(predictions)):

        image_url = predictions['url'][i]
        filename = predictions['filename'][i]

        with open(os.path.join(path,filename), 'wb') as handler: 
            handler.write(requests.get(image_url).content)  

    threads = []

    for i in range(0, 30):
        
        # image_path, estimated_excution_time, sledge_url, rps, i
        threads.append(Thread(target=worker, args=(os.path.join(path,predictions['filename'][i]), float(predictions[predictions_index][i]) * pow(10, 6), sledge_url, requests_per_second, total_seconds, i)))
        threads[i].start()

    for t in threads:
        t.join()

    all_data = pd.DataFrame(columns=("filename", "status_code"))

    for i in range(0, 30):
        cur_csv = pd.read_csv("result{}.csv".format(i))
        for j in range(0, len(cur_csv)):
            all_data.loc[len(all_data.index)] = [predictions['filename'][i], cur_csv['status-code'][j]]
    
    all_data.to_csv('results.csv')


    for i in range(0, len(predictions)):
        filename = predictions['filename'][i]
        os.remove(os.path.join(path,filename))
        os.remove("result{}.csv".format(i))
        