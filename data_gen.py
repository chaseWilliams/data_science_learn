import numpy as np
import requests as http
import pprint

library_data = {}
token = 'BQDTWMPn1EMRk-u2DN74mEK-X-BxGTM3oqxRTFgsHyGRiCt3Z3ie_ilygpXK0RULAt8iN1iOgslhBFW4hlrwI_Pxy-kRX1XMmNKXF63Y63jZfSnPLuRhm55qT3NnOcSfp38kIxroIs5A4lBkkCnCB0EawFtdZ5MB'
api_base = 'https://api.spotify.com/v1'
api_library = api_base + '/me/tracks'
api_artists = api_base + '/artists'

def add_artists(artists):
    for artist in artists:
        artist_name = artist['name']
        if artist_name in library_data:
            library_data[artist_name]['count'] += 1
        else:
            library_data[artist_name] = {'id': artist['id'], 'count': 1}

# takes a list of spotify IDs (max - 50)
def determine_genres(ids):
    string = ''
    genres = [] # each elem's index corresponds with ids' index
    for id in ids:
        string += id
    response = http.get(api_artists + '?ids=' + string, headers={'Authorization': 'Bearer ' + token})
    json = response.json()
    for artist in json['artists']:
        genres.append(artist['genres'])
    return genres

# token is temporary

response = http.get(api_library + '?limit=1', headers={'Authorization': 'Bearer ' + token})
result = response.json()
total_songs = result['total']
# adds an initial song
add_artists(result['items'][0]['track']['artists'])

for offset in list(range(1, total_songs, 50)):
    print('doing offset' + str(offset))
    batch = http.get(api_library + '?limit=50&offset=' + str(offset), headers={'Authorization': 'Bearer ' + token})
    batch_json = batch.json()
    for track in batch_json['items']:
        add_artists(track['track']['artists'])

## At this point, library_data has all the artists, their respective counts,
## and their Spotify ID.

## Figure out what genre the artists are from

keys = library_data.keys()
for index in list(range(1, total_songs, 50)):
    sub_arr = keys[:index]
    string = ''
    for key in sub_arr:
        string += library_data[key]['id']
    string.index(0, str)


## visualize result
pprint.pprint(library_data)
print(len(library_data))
