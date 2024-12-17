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
bsky_topic_df = pd.read_csv("../data/bsky_df_id_topic.csv")

bsky_threads = []
from tqdm.auto import tqdm
for bsky_id in tqdm(bsky_topic_df['id']):
    if "at" not in bsky_id:
        break
    thread = get_thread(bsky_id)
    bsky_threads.append(thread) 

# Save the threads to a file
with open("../data/bsky_threads.json", "w") as f:
    json.dump(bsky_threads, f)