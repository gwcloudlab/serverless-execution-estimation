import predictor
import pandas as pd

database = pd.read_csv('analysis_database.csv', delimiter=',')
database['filename'] = database['url'].map(lambda url : "images/" + url.split('/')[-1])
images = pd.DataFrame(index=range(len(database)), columns=('url', 'filename','estimated_time_small', 'estimated_time_medium', 'estimated_time_large'))

for i in range(0, len(database)):
    images['estimated_time_small'][i] = predictor.predict("resize_small", database['file_size'][i] ,database['sizex'][i],database['sizey'][i])[0]
    images['estimated_time_medium'][i] = predictor.predict("resize_medium", database['file_size'][i] ,database['sizex'][i],database['sizey'][i])[0]
    images['estimated_time_large'][i] = predictor.predict("resize_large", database['file_size'][i] ,database['sizex'][i],database['sizey'][i])[0]
    images['filename'][i] = database['filename'][i]
    images['url'][i] = database['url'][i]
images.to_csv('predictions.csv', index=False)

