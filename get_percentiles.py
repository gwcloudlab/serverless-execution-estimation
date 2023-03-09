import pandas as pd
import numpy as np

database = pd.read_csv('database3.csv')
percentiles = pd.DataFrame(columns = ('small', 'medium', 'large'), index=range(10))

resize_small = np.array(database['response_time_resize_small'])
resize_medium = np.array(database['response_time_resize_medium'])
resize_large = np.array(database['response_time_resize_large'])


for i in range(10, 110, 10):
   percentiles['small'][(i/10)-1] = np.percentile(resize_small, i)
   percentiles['medium'][(i/10)-1] = np.percentile(resize_small, i)
   percentiles['large'][(i/10)-1] = np.percentile(resize_small, i)

percentiles.to_csv('percentiles.csv', index=False)