import dota2api
import requests
import json
import pickle
import pandas as pd

matches_array = pickle.load(open("matches.p", "rb" ))


data = pd.DataFrame([], index = [ 'radiant1', 'radiant2', 'radiant3', 'radiant4','radiant5', 'opponent1', 'opponent2', 'opponent3', 'opponent4', 'opponent5','outcome'])

print(data.head())
for match in matches_array:
    players_array= match["players"]
    radiant = []
    dire = []
    outcome = int(match['radiant_win'])
    for player in players_array:
        if player['isRadiant'] == True:
            radiant.append(player['hero_id'])
        else:
            dire.append(player['hero_id'])

    radiant = [int(x) for x in radiant]
    dire = [int(x) for x in dire]

    data = data.append(pd.Series(radiant + dire + [outcome], index = ['radiant1', 'radiant2', 'radiant3', 'radiant4','radiant5', 'opponent1', 'opponent2', 'opponent3', 'opponent4', 'opponent5','outcome']), ignore_index=True)

data = data.dropna()
data.to_csv('out.csv')
print(len(data))
print(data.head(15))