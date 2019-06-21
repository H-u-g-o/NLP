import requests
import base64
#http://www.trumptwitterarchive.com/archive
client_key = 'Ln0syzM5FdGI4JIWnECaTpMms'
client_secret = 'NoUAYRoPCyWrIaLCOk6S6B8ag65WOAKg58gTyAft5MDQiphJfh'

def init_api() :
    key_secret = '{}:{}'.format(client_key, client_secret).encode('ascii')
    b64_encoded_key = base64.b64encode(key_secret)
    b64_encoded_key = b64_encoded_key.decode('ascii')

    base_url = 'https://api.twitter.com/'
    auth_url = '{}oauth2/token'.format(base_url)

    auth_headers = {
        'Authorization': 'Basic {}'.format(b64_encoded_key),
        'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'
    }

    auth_data = {
        'grant_type': 'client_credentials'
    }

    auth_resp = requests.post(auth_url, headers=auth_headers, data=auth_data)

    access_token = auth_resp.json()['access_token']

    return access_token

def main() :
    access_token = init_api()
    headers = {
        'Authorization': 'Bearer {}'.format(access_token)
    }

    #r = requests.get("https://api.twitter.com/1.1/favorites/list.json?count=200&user_id=25073877&exclude_replies=true?tweet_mode=extended", headers=headers)
    #r = requests.get("https://api.twitter.com/1.1/statuses/list.json?user_id=25073877&count=200&exclude_replies=true?include_rts=1")
    r = requests.get("https://api.twitter.com/1.1/tweets/search/fullarchive/trumpisation.json")
    print (r.json())

if __name__ == '__main__':
    main()