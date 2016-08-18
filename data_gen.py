import numpy as np
import requests as http
import pprint

api_base = 'https://api.spotify.com/v1'
api_library = api_base + '/me/tracks'
# regenerate token
token = ''
response = http.get(api_library + '?limit=1', headers={'Authorization': 'Bearer ' + token})
result = response.json()
print(type(result))
#pprint.pprint(result)
print(result['items'][0]['track']['artists'][0]['id']) #don't forget about multiple artists
