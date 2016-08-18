import numpy as np
import requests as http
import pprint

library_data = {}
token = 'BQBOCBoZ9vzHbQl01hfc_eOV9zFRqyY90ge1mSYYHU_BIrLuSIVFXLJ7LpjmR_m2F3KZdvP8Xp4ovmOEuSgD3q0q2elxG__PkoRzrGUm0nppn68Fn9e-yGC6pa8LSH91lXlzME5Lz5n-OwTd-hHew4kUXOWGbI5m'
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
        string += id + ','
    string = string.rstrip(',')
    print("\n\n\n" + string)
    artists_get_result = http.get(api_artists + '?ids=' + string, headers={'Authorization': 'Bearer ' + token})
    json = artists_get_result.json()
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
keys = list(keys)
artists_genres = []
total_artists = len(library_data)
range_ = range(1, total_artists, 50)
print(list(range_))
for index in list(range_):
    if index + 50 > total_artists:
        final_index = total_artists
    else:
        final_index = index + 50

    sub_arr = keys[index:final_index]
    id_arr = []
    for key in sub_arr:
        id_arr.append(library_data[key]['id'])
    artists_genres.append( determine_genres(id_arr) )

## Analyze artist_genres for genre counts

genre_count = {}
for specific_artist in artists_genres:
    for genres in specific_artist:
        for genre in genres:
            print(type(genre))
            if genre not in genre_count:
                genre_count[genre] = 1
            else:
                genre_count[genre] += 1



## visualize result

pprint.pprint(library_data)
pprint.pprint(genre_count)
print(len(library_data))
print(total_artists)
genre_keys = genre_count.keys()
count = 0
for key in genre_keys:
    count += genre_count[key]['count']
print(count)