import dota2api
import requests
import json
import pickle
import pandas as pd

matches_array = pickle.load(open("matches.p", "rb" ))


data = pd.DataFrame([], index = ['hero', 'allied1', 'allied2', 'allied3', 'allied4', 'opponent1', 'opponent2', 'opponent3', 'opponent4', 'opponent5','outcome'])

print(data.head())
for match in matches_array:
    players_array= match["players"]
    radiant = []
    dire = []
    for player in players_array:
        if player['isRadiant'] == True:
            radiant.append(player['hero_id'])
        else:
            dire.append(player['hero_id'])

    radiant = [int(x) for x in radiant]
    dire = [int(x) for x in dire]
    for player in players_array:
        hero = int(player['hero_id'])
        allied_heroes = []
        opponent = []
        outcome = int(player['win'])
        
        if player['isRadiant']:
            allied_heroes = [x for x in radiant if x != hero]
            opponent = [x for x in dire]
        else:
            allied_heroes = [x for x in dire if x != hero]
            opponent = [x for x in radiant]

        data = data.append(pd.Series([hero] + allied_heroes + opponent + [outcome], index = ['hero', 'allied1', 'allied2', 'allied3', 'allied4', 'opponent1', 'opponent2', 'opponent3', 'opponent4', 'opponent5','outcome']), ignore_index=True)




data = data.dropna()
data.to_csv('out.csv')
print(data.head(15))