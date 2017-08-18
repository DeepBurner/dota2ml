import dota2api
import requests
import json
import pickle
import pandas as pd
import numpy as np

data = pd.read_csv('out.csv')

allied_slice = data[['radiant1' , 'radiant2', 'radiant3', 'radiant4', 'radiant5']]

max = 114
print(max)

# print(allied_slice.head())

colnames = ['AHero_{}.0'.format(x) for x in range(0,max+1)]
a_vectorised = pd.DataFrame(0, index=range(len(data)), columns=colnames)

for i in ['radiant1' , 'radiant2', 'radiant3', 'radiant4', 'radiant5']:
    _asd = pd.get_dummies(data[[i]].applymap(str), prefix='AHero').astype(np.int64)
    a_vectorised = a_vectorised.add(_asd, fill_value=0)

a_vectorised = a_vectorised.astype(np.int64)

colnames = ['OHero_{}.0'.format(x) for x in range(0,max+1)]
o_vectorised = pd.DataFrame(0, index=range(len(data)), columns=colnames)

for i in ['opponent1', 'opponent2', 'opponent3', 'opponent4', 'opponent5']:
    _asd = pd.get_dummies(data[[i]].applymap(str), prefix='OHero').astype(np.int64)
    o_vectorised = o_vectorised.add(_asd, fill_value=0)


data_vectorised = pd.concat([a_vectorised, o_vectorised, data[['outcome']]], axis=1)

print(data_vectorised.describe())

data_vectorised.to_csv('worked.csv')