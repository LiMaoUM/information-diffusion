import pandas as pd
#Bluesky thread api
import requests
import json
import os

def get_thread(thread_id):
    url = f"https://public.api.bsky.app/xrpc/app.bsky.feed.getPostThread"
    params = {
        "uri":thread_id,
        'depth': 1000
    }
    response = requests.get(url, params=params)
    return response.json()
def get_repost(post_id):
    url = f"https://public.api.bsky.app/xrpc/app.bsky.feed.getRepostedBy"
    params = {
        "uri":post_id,
        "limit": 100
    }
    response = requests.get(url, params=params)
    return response

def get_userinfo(user_id):
    url = f"https://public.api.bsky.app/xrpc/app.bsky.actor.getProfile"
    params = {
        'actor': user_id
    }
    response = requests.get(url, params=params)
    return response.json()

#bsky_topic_df = pd.read_csv("../data/bsky_df_id_topic.csv")

#bsky_threads = []
#from tqdm.auto import tqdm
#for bsky_id in tqdm(bsky_topic_df['id']):
    #if "at" not in bsky_id:
        #break
    #thread = get_thread(bsky_id)
    #bsky_threads.append(thread) 

## Save the threads to a file
#with open("../data/bsky_threads.json", "w") as f:
    #json.dump(bsky_threads, f)

with open("../data/bsky_follows.json") as f:
    bsky_follow = json.load(f)

from collections import defaultdict
from itertools import chain

original_list = bsky_follow

# Use a defaultdict to store sets of DIDs.
merged = defaultdict(set)

# chain.from_iterable(...) flattens out the "dict.items()" across the list
for key, records in chain.from_iterable(item.items() for item in original_list):
    # 'records' is the list of dicts. We update the set with the "did" values.
    merged[key].update(r["did"] for r in records)

# Convert to a regular dict if desired:
merged_dict = dict(merged)

author_list = list(merged_dict.keys())
author_dict = {}
for author in author_list:
    user_info = get_userinfo(author)
    author_dict[author] = user_info
    with open("../data/bsky_author_info.json", "w") as f:
        json.dump(author_dict, f) 

with open("../data/bsky_author_info.json", "w") as f:
    json.dump(author_dict, f)