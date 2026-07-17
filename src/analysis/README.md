# Zenodo Dataset Description

## Dataset Title
Information Diffusion on Decentralized Social Media: BlueSky vs. Truth Social - Posts, Reposts, and Follow Networks

## Overview

This dataset contains **posts, repost/reblog activities, and user follow networks** collected from two decentralized social media platforms: **BlueSky** and **Truth Social**. The data represents information cascades, user engagement patterns, and social network structures during the 2024-2025 period.

## Data Contents

### 1. **Posts & Reposts Data**

#### BlueSky Posts with Repost Activities
- **File**: `bsky_reposts_new.json`
- **Format**: JSON array
- **Records**: ~[X] posts
- **Fields per record**:
  - `_id`: Unique post URI
  - `author`: Author metadata (DID, handle, display name)
  - `record`: Post content and metadata
    - `text`: Post text content
    - `createdAt`: Timestamp
    - `langs`: Language tags
    - `reply`: Parent/root post information (if reply)
  - `repostCount`: Number of reposts
  - `replyCount`: Number of replies
  - `likeCount`: Number of likes
  - `indexedAt`: Crawl timestamp
  - `reposts`: (optional) Repost user information

#### Truth Social Posts with Reblog Activities
- **File**: `ts_threads_withReblogs.json`
- **Format**: JSON array (Mastodon-compatible format)
- **Records**: ~[X] posts
- **Fields per record**:
  - `_id`: Unique post ID
  - `content`: HTML-formatted post content
  - `created_at`: Timestamp
  - `account`: Author metadata (ID, username, display name, followers count)
  - `in_reply_to_id`: Parent post ID (if reply)
  - `replies_count`: Number of replies
  - `reblogs_count`: Number of reblogs
  - `favourites_count`: Number of likes
  - `visibility`: Post visibility level
  - `sensitive`: Content warning flag
  - `language`: Language code

### 2. **Follow Network Data**

#### BlueSky Follow Graph
- **File**: `bsky_follows.json`
- **Format**: JSON array of dictionaries
- **Structure**: `[{ "follower_did": [ {followee_metadata}, ...] }, ...]`
- **Contains**:
  - Follower DIDs (decentralized identifiers)
  - Followee metadata: DID, handle, display name, description, followers count, avatar, account creation date

#### Truth Social Follow Graph
- **File**: `ts_user_following_map.json`
- **Format**: JSON dictionary
- **Structure**: `{ "user_id": [followee_id, ...], ... }`
- **Contains**:
  - Source user ID → list of followed user IDs
  - Direct following relationships (no additional metadata in this file)

### 3. **Ideology Labels** (Optional Supplementary Data)

#### Post-level Ideology Classification
- **Files**: 
  - `bsky_post_to_label.json`: Maps post text → ideology label (left/center/right)
  - `ts_post_to_label.json`: Maps post text → ideology label (left/center/right)
- **Labels**: left, center, right, lean-left, lean-right

#### Author-level Ideology Distribution
- **Files**:
  - `bsky_author_ideology_portion.json`: Author → {left: ratio, center: ratio, right: ratio}
  - `ts_author_ideology_portion.json`: Similar structure for Truth Social

### 4. **Topic Mappings** (Optional Supplementary Data)

#### Post-to-Topic Assignment
- **Files**:
  - `bsky_df_id_topic.csv`: Columns: [id, topic_label, topic_number]
  - `ts_df_id_topic.csv`: Similar structure for Truth Social
- **Topics**: Political discussion topics extracted via BERTopic model

---

## Data Characteristics

| Metric | BlueSky | Truth Social |
|--------|---------|--------------|
| Posts | ~[X] | ~[X] |
| Authors | ~[X] | ~[X] |
| Follow edges | ~[X] | ~[X] |
| Date range | 2024-2025 | 2024-2025 |
| Language(s) | Primarily English | Primarily English |

---

## Collection Methodology

### BlueSky Data
- **API Used**: Bluesky Public API (`app.bsky.feed.getPostThread`, `app.bsky.graph.getFollows`)
- **Collection Method**: 
  - Recursive thread scraping (depth-first traversal)
  - Follow graph enumeration per user
- **Timestamp**: Post creation time (`createdAt`) and API index time (`indexedAt`)

### Truth Social Data
- **API Used**: Mastodon-compatible API (Truth Social runs Mastodon fork)
- **Collection Method**:
  - Thread timeline collection
  - Reblog activity tracking
  - User follow graph retrieval
- **Timestamp**: Post creation time (`created_at`)

---

## Data Format & Structure

### JSON Structure (Posts)

**BlueSky example**:
```json
{
  "_id": "at://did:plc:xxx/app.bsky.feed.post/xxxxx",
  "author": {
    "did": "did:plc:xxx",
    "handle": "user.bsky.social",
    "displayName": "User Name"
  },
  "record": {
    "text": "Post content here",
    "createdAt": "2024-12-15T10:30:00.000Z",
    "langs": ["en"],
    "reply": {
      "parent": {"uri": "at://..."},
      "root": {"uri": "at://..."}
    }
  },
  "repostCount": 5,
  "replyCount": 12,
  "likeCount": 45,
  "indexedAt": "2024-12-15T10:31:00.000Z"
}