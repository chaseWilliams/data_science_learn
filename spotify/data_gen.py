import numpy as np
import requests as http
import pprint
import matplotlib.pyplot as plt

library_data = {}
token = 'BQBUaD3_sEXuoKB2cJxHUgSi0jUqHqp5H1EZZMvryJHg0JJ7c4PPPambwDEt1QOWi7JQCawGUIN1wPn40J3N0ZZuaQv_aom8g4Tb8ZjNKCkmW8GuGGVGvHqchNwIgghbksEAa_tHQNKfbQ7yx9w'
api_base = 'https://api.spotify.com/v1'
api_library = api_base + '/me/tracks'
api_artists = api_base + '/artists'
edm_words = ['edm' , 'big' , 'beat' , 'step' , 'dance' , 'house' , 'progressive' , 'trance', 'trap']
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
    artists_get_result = http.get(api_artists + '?ids=' + string, headers={'Authorization': 'Bearer ' + token})
    json = artists_get_result.json()
    for artist in json['artists']:
        genres.append(artist['genres'])
    return genres

def str_check(words, word):
    if len(words) > 0:
        if words[0] in word:
            return True
        else:
            return str_check(words[1:], word)
    else:
        return False

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



## textual visualization result

pprint.pprint(library_data)
pprint.pprint(genre_count)
print(len(library_data))
print(total_artists)
genre_keys = genre_count.keys()
generic_genre_count = {
    'pop': 0,
    'edm': 0,
    'rap': 0,
    'alternative': 0,
    'other': 0
}
for key in genre_keys:
    if str_check(edm_words, key):
        generic_genre_count['edm'] += 1
    elif 'rap' in key:
        generic_genre_count['rap'] += 1
    elif 'alternative' in key:
        generic_genre_count['alternative'] += 1
    elif 'pop' in key:
        generic_genre_count['pop'] += 1
    else:
        generic_genre_count['other'] += 1

print("the number of genres identified is " + str(len(genre_keys)))
print("number of songs" + str(total_songs))
print("number of artists" + str(total_artists))

pprint.pprint(generic_genre_count)

labels = ['Alternative', 'EDM', 'Other', 'Pop', 'Rap']
sizes = [
    generic_genre_count['alternative'],
    generic_genre_count['edm'],
    generic_genre_count['other'],
    generic_genre_count['pop'],
    generic_genre_count['rap']
]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'blue']
explode = [0, 0.2, 0, 0, 0]
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
#plt.show()
count = 0
for key in library_data.keys():
    if library_data[key]['count'] >= 3:
        print(key)
        print(library_data[key]['count'])
        count += 1
print(count)
print(float(count) / float(total_artists))