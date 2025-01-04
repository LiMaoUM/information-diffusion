import requests
import json
def fetch_flower_information(base_url, author, limit=100):
    """
    Fetches flower information from the API using pagination.

    Parameters:
        base_url (str): The API endpoint.
        author (str): The author ID.
        limit (int): Number of items to fetch per request.

    Returns:
        list: A list of all flower information retrieved from the API.
    """
    all_flower_info = []
    params = {
        "actor": author,
        "limit": limit
    }

    while True:
        response = requests.get(base_url, params=params)

        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            break

        data = response.json()
        
        # Add the flower information from the current response
        all_flower_info.extend(data.get("follows", []))
        #all_flower_info.extend(data.get("followers", []))

        # Check if there's a cursor for the next page
        cursor = data.get("cursor")
        if not cursor:
            break

        # Update the params with the new cursor
        params["cursor"] = cursor

    return all_flower_info

if __name__ == "__main__":
    base_url = "https://public.api.bsky.app/xrpc/app.bsky.graph.getFollows"
    from tqdm.auto import tqdm
    with open("../data/bsky_reposts.json", "r") as f:
        bsky_threads = json.load(f)
    all_authors = set()
    for i in tqdm(bsky_threads):
        all_authors.add(i['author']['did'])
        if i.get('reposts',None) and i['reposts'] != []:
            for j in i['reposts']:
                all_authors.add(j['did'])
    print(f"Total authors: {len(all_authors)}")
    authors = list(all_authors)
    all_flower_info = []
    for author in tqdm(authors):
        all_flower_info.append({author: fetch_flower_information(base_url, author)})
    with open("../data/bsky_follows.json", "w") as f:
        json.dump(all_flower_info, f)