import dota2api
import requests
import json
import pickle
import pandas as pd

# api = dota2api.Initialise("8FFEFEFDD9722E8BEDE06EE298493AE3")

# match = api.get_match_details(match_id=1000193456)
# print(match)

req = requests.get("https://api.opendota.com/api/promatches")

matches_array = req.json()

print(len(matches_array))

matches = []

for i in matches_array:
    try:
        print("https://api.opendota.com/api/matches/%i" % i['match_id'])
        _url = "https://api.opendota.com/api/matches/%i" % i['match_id']
        matches.append(requests.get(_url).json())
    except:
        pass


print(len(matches))

req2 = requests.get("https://api.opendota.com/api/publicmatches")

matches_array = req.json()

print(len(matches_array))

for i in matches_array:
    try:
        print("https://api.opendota.com/api/matches/%i" % i['match_id'])
        _url = "https://api.opendota.com/api/matches/%i" % i['match_id']
        matches.append(requests.get(_url).json())
    except:
        pass

pickle.dump(matches, open("matches.p", "wb" ))

# _d = {'hero': [], 'allied1': [], 'allied2': [], 'allied3': [], 'allied4': [], 'opponent1':[], 'opponent2':[], 'opponent3':[], 'opponent4':[], 'opponent5':[], 'outcome':[]}
# data = pd.DataFrame(_d)

# for match in matches:
#     print(match)
#     players_array= match["players"]
#     radiant = []
#     dire = []
#     for player in players_array:
#         if player['isRadiant'] == True:
#             radiant.append(player['hero_id'])
#         else:
#             dire.append(player['hero_id'])

#     for player in players_array:
#         hero = player['hero_id']
#         allied_heroes = []
#         opponent = []
#         outcome = player['win']

#         if player['isRadiant'] == True:
#             for plyr in radiant not in [hero]:
#                 allied_heroes.append(plyr)
#             for opp in dire:
#                 opponent.append(opp)
#         else:
#             for plyr in dire not in [hero]:
#                 allied_heroes.append(plyr)
#             for opp in dire:
#                 opponent.append(opp)

#         data.append(pd.Series(hero + allied_heroes + opponent + outcome), ignore_index = True)

#         print(data.head())