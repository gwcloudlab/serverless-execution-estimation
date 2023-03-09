import requests
import time
import pandas as pd
import os

pd.options.mode.chained_assignment = None  

database = pd.read_csv('database_2.csv', delimiter=',')
database = database[0:round(len(database)/2)]
# database['urlName'] = database['url'].map(lambda url : url.split('/')[-1])
database["response_time_resize_large"] = ""




# Image_resize, image_classification, image format transform
def generateData():
  path =  os.path.join(os.path.dirname(os.path.realpath(__file__)), 'images')
  if not os.path.exists(path):
      os.mkdir(path)

  for x in range(0, len(database)):

    if database["response_time_resize_large"][x] != "":
      continue
   
    image_url = database['url'][x]

    with open(os.path.join(path, image_url.split('/')[-1]), 'wb') as handler: 
        handler.write(requests.get(image_url).content)  

    url = 'http://localhost:10000/resize_large/'

    with open(os.path.join(path, image_url.split('/')[-1]), 'rb') as f:
      file_data = f.read()
    

    try:
      response = requests.post(url, data=file_data)

      database["response_time_resize_large"][x] = response.elapsed.total_seconds()
      
    except Exception as err:
      print(err)
      database["response_time_resize_large"][x] = pd.NA

       
      
    os.remove(os.path.join(path, image_url.split('/')[-1]))
  database.to_csv('database_3.csv', index=False)
  
generateData()