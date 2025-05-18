import os
import json
import pandas as pd
import argparse

def convert_text(text, rt_text):
    """Convert the text of a tweet to include the text of the retweet."""
    return text.split(':')[0] + ': ' + rt_text if text.startswith('RT @') else text

def convert_dataframe(tweets_data):
    """Extract relevant information from the tweets and convert it to a DataFrame."""
    tweets_list = []
    for tweet in tweets_data:
        # tweet basic information
        tweet_dict = {'user': tweet['user']['screen_name'],
                      'user_name': tweet['user']['name'],
                      'description': tweet['user']['description'],
                      'location': tweet['user']['location'],
                      'protected': tweet['user']['protected'],
                      'verified': tweet['user']['verified'],
                      'created': tweet['user']['created_at'],
                      'followers': tweet['user']['followers_count'],
                      'friends': tweet['user']['friends_count'],
                      'listed': tweet['user']['listed_count'],
                      'favourites': tweet['user']['favourites_count'],
                      'statuses': tweet['user']['statuses_count'],
                      'default_profile': tweet['user']['default_profile'],
                      'default_image': tweet['user']['default_profile_image'],
                      'time': tweet['created_at'],
                      'id': tweet['id'],
                      'text': tweet['full_text'],
                      'retweet_count': tweet['retweet_count'],
                      'favourite_count': tweet['favorite_count'],
                      'reply': tweet['in_reply_to_screen_name'],
                      'language': tweet['lang'], 
                      'entity_image_url': '',
                      'extended_entity_image_urls': [],
                      'place': '',
                      'latitude': '',
                      'longitude': ''}
        
        # place information
        if 'place' in tweet:
            tweet_dict['place'] = tweet['place']
        if 'coordinates' in tweet and tweet['coordinates'] is not None:
            tweet['longitude'] = tweet['coordinates']['coordinates'][0]
            tweet['lattitude'] = tweet['coordinates']['coordinates'][1]
        
        # image information
        if 'entities' in tweet and 'media' in tweet['entities']:
            for media in tweet['entities']['media']:
                if media['type'] == 'photo':
                    tweet_dict['entity_image_url'] = media['media_url_https']
                    break
        if 'extended_entities' in tweet and 'media' in tweet['extended_entities']:
            for media in tweet['extended_entities']['media']:
                if media['type'] == 'photo':
                    tweet_dict['extended_entity_image_urls'].append(media['media_url_https'])

        # retweet information
        if 'retweeted_status' in tweet:
            tweet_dict['rt_user'] = tweet['retweeted_status']['user']['screen_name']
            tweet_dict['rt_user_name'] = tweet['retweeted_status']['user']['name']
            tweet_dict['rt_description'] = tweet['retweeted_status']['user']['description']
            tweet_dict['rt_location'] = tweet['retweeted_status']['user']['location']
            tweet_dict['rt_protected'] = tweet['retweeted_status']['user']['protected']
            tweet_dict['rt_verified'] = tweet['retweeted_status']['user']['verified']
            tweet_dict['rt_created'] = tweet['retweeted_status']['user']['created_at']
            tweet_dict['rt_followers'] = tweet['retweeted_status']['user']['followers_count']
            tweet_dict['rt_friends'] = tweet['retweeted_status']['user']['friends_count']
            tweet_dict['rt_listed'] = tweet['retweeted_status']['user']['listed_count']
            tweet_dict['rt_favourites'] = tweet['retweeted_status']['user']['favourites_count']
            tweet_dict['rt_statuses'] = tweet['retweeted_status']['user']['statuses_count']
            tweet_dict['rt_default_profile'] = tweet['retweeted_status']['user']['default_profile']
            tweet_dict['rt_default_image'] = tweet['retweeted_status']['user']['default_profile_image']
            tweet_dict['rt_time'] = tweet['retweeted_status']['created_at']
            tweet_dict['rt_id'] = tweet['retweeted_status']['id']
            tweet_dict['rt_text'] = tweet['retweeted_status']['full_text']
            tweet_dict['rt_retweet_count'] = tweet['retweeted_status']['retweet_count']
            tweet_dict['rt_favourite_count'] = tweet['retweeted_status']['favorite_count']            
            tweet_dict['rt_reply'] = tweet['retweeted_status']['in_reply_to_screen_name']
            tweet_dict['rt_language'] = tweet['retweeted_status']['lang']
       
        # quote information
        if 'quoted_status' in tweet:
            tweet_dict['qt_user'] = tweet['quoted_status']['user']['screen_name']
            tweet_dict['qt_user_name'] = tweet['quoted_status']['user']['name']
            tweet_dict['qt_description'] = tweet['quoted_status']['user']['description']
            tweet_dict['qt_location'] = tweet['quoted_status']['user']['location']
            tweet_dict['qt_protected'] = tweet['quoted_status']['user']['protected']
            tweet_dict['qt_verified'] = tweet['quoted_status']['user']['verified']
            tweet_dict['qt_created'] = tweet['quoted_status']['user']['created_at']
            tweet_dict['qt_followers'] = tweet['quoted_status']['user']['followers_count']
            tweet_dict['qt_friends'] = tweet['quoted_status']['user']['friends_count']
            tweet_dict['qt_listed'] = tweet['quoted_status']['user']['listed_count']
            tweet_dict['qt_favourites'] = tweet['quoted_status']['user']['favourites_count']
            tweet_dict['qt_statuses'] = tweet['quoted_status']['user']['statuses_count']
            tweet_dict['qt_default_profile'] = tweet['quoted_status']['user']['default_profile']
            tweet_dict['qt_default_image'] = tweet['quoted_status']['user']['default_profile_image']
            tweet_dict['qt_time'] = tweet['quoted_status']['created_at']
            tweet_dict['qt_id'] = tweet['quoted_status']['id']
            tweet_dict['qt_text'] = tweet['quoted_status']['full_text']
            tweet_dict['qt_retweet'] = tweet['quoted_status']['retweet_count']
            tweet_dict['qt_favourite_count'] = tweet['quoted_status']['favorite_count']    
            tweet_dict['qt_reply'] = tweet['quoted_status']['in_reply_to_screen_name']
            tweet_dict['qt_language'] = tweet['quoted_status']['lang']  
        
        tweets_list.append(tweet_dict)
    df = pd.DataFrame(tweets_list)
    return df


def read_json(filepath):
    """Read a JSON file and return a list of dictionaries."""
    tweets_data = list()
    with open(filepath, 'r') as tweets_file:
        for line in tweets_file:
            try:
                tweet = json.loads(line)
                tweets_data.append(tweet)
            except:
                continue
    return tweets_data


def main(read_folder, save_folder):
    """Convert the tweet data to a DataFrame and save it to a JSON file."""
    files = os.listdir(read_folder)
    files = sorted(files)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for file in files:
        read_filepath = os.path.join(read_folder, file)
        tweets_data = read_json(read_filepath)
        df = convert_dataframe(tweets_data)
        df['text'] = df.apply(lambda row: convert_text(str(row['text']), str(row['rt_text'])), axis=1)
        save_filepath = os.path.join(save_folder, file)
        df.to_json(save_filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract tweet information.')
    parser.add_argument('read_folder', type=str, help='The folder to read tweet data from.')
    parser.add_argument('save_folder', type=str, help='The folder to save the converted tweet data.')
    args = parser.parse_args()
    main(args.read_folder, args.save_folder)