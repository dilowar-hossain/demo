import os
import pandas as pd
from googleapiclient.discovery import build

#  API key
API_KEY = 'AIzaSyDfioc28FiF1nYVELHdrLXzkutP0Gnsbfc'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

def get_authenticated_service():
    return build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

def get_video_data(youtube, query, max_results=50):
    # Search for videos matching the query
    search_response = youtube.search().list(
        q=query,
        part='id,snippet',
        maxResults=max_results,
        type='video'
    ).execute()
    
    videos = []
    
    for search_result in search_response.get('items', []):
        video_id = search_result['id']['videoId']
        video_title = search_result['snippet']['title']
        video_description = search_result['snippet']['description']
        
        videos.append({
            'video_id': video_id,
            'video_title': video_title,
            'video_description': video_description
        })
    
    return videos

def get_video_statistics(youtube, video_ids):
    # Get statistics for each video
    response = youtube.videos().list(
        part='statistics',
        id=','.join(video_ids)
    ).execute()
    
    stats = []
    
    for video in response.get('items', []):
        video_id = video['id']
        view_count = video['statistics'].get('viewCount', 0)
        like_count = video['statistics'].get('likeCount', 0)
        comment_count = video['statistics'].get('commentCount', 0)
        
        stats.append({
            'video_id': video_id,
            'view_count': view_count,
            'like_count': like_count,
            'comment_count': comment_count
        })
    
    return stats

def merge_data(video_data, video_stats):
    # Convert lists to dataframes
    df_videos = pd.DataFrame(video_data)
    df_stats = pd.DataFrame(video_stats)
    
    # Merge dataframes on video_id
    merged_df = pd.merge(df_videos, df_stats, on='video_id')
    
    return merged_df

def main():
    youtube = get_authenticated_service()
    
    # Define the search query
    query = 'python programming'
    
    # Get video data
    video_data = get_video_data(youtube, query)
    
    # Get video statistics
    video_ids = [video['video_id'] for video in video_data]
    video_stats = get_video_statistics(youtube, video_ids)
    
    # Merge data
    data = merge_data(video_data, video_stats)
    
    # Save data to CSV
    data.to_csv('youtube_data.csv', index=False)
    print('Data saved to youtube_data.csv')

if __name__ == '__main__':
    main()
