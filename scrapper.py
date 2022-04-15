import requests
import time
import pandas as pd
import os

database = pd.read_csv('https://gitlab.com/lenapster/faascache/-/raw/master/machine-learning/data/features_image.csv', delimiter=';')
database['urlName'] = database['url'].map(lambda url : url.split('/')[-1])
database["exec_time"] = ""



def getUrls():
    path =  os.path.join(os.path.dirname(os.path.realpath(__file__)), 'images')
    if not os.path.exists(path):
      os.mkdir(path)
    for x in range(25674, len(database['url'])):
      url = database['url'][x]
      with open(os.path.join(path, url.split('/')[-1]), 'wb') as handler: 
        handler.write(requests.get(url).content)
  
# Image_resize, image_classification, image format transform
def generateData():
  path =  os.path.join(os.path.dirname(os.path.realpath(__file__)), 'images')
  for x in range(0,1):
    if database["exec_time"][x] != "":
      continue
      
    url = 'http://localhost:10000/'
    my_img = {'image': open(os.path.join(path,database['urlName'][x]), 'rb')}
    try:
      response = requests.post(url, files=my_img)
      print(response)
      database["exec_time_0"][x] = response.elapsed.total_seconds()
      
    except Exception as err:
       database["exec_time"][x] = pd.NA
  print(database)