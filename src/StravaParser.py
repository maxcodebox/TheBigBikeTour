import os
import re
import requests
from bs4 import BeautifulSoup
import pickle
import numpy as np
from geopy.distance import geodesic
from stravalib import Client
import trips as tr
import tokens

def get_activity_photos_from_web(activity_id, size = 5000):
    # https://communityhub.strava.com/t5/developer-discussions/download-all-photos-of-my-own-activities/m-p/11262
    # Construct the URL manually
    url = f"https://www.strava.com/api/v3/activities/{activity_id}/photos?size={size}"

    # Headers including the OAuth token for authentication
    headers = {
        'Authorization': f"Bearer {tokens.ACCESS_TOKEN}"
    }

    # Making the GET request to Strava API
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        photos = response.json()  # The photos data in JSON format
        return photos
    else:
        print("Error:", response.status_code, response.text)

def update_all():
    client = Client(access_token=tokens.ACCESS_TOKEN)
    activity_ids = [trip for trip, _ in tr.trip_dicts.items()]
    for activity_id in activity_ids:
        print(f"Updating {activity_id}")
        activity_dict = import_activity(activity_id, client, reload=True)

def import_collection(collection_name, reload = False):
    
    if 'sort_by' in tr.collection_dict[collection_name].keys():
        sort_by = tr.collection_dict[collection_name]['sort_by']
    else:
        sort_by = 'location'
    
    client = Client(access_token=tokens.ACCESS_TOKEN)
    
    activity_dicts = {}
    if 'start_coords' in tr.collection_dict[collection_name].keys():
        start_coords = tr.collection_dict[collection_name]['start_coords']
    else:
        start_coords = (90,0)
    activity_ids = []
    
    for activity_id, activity_dict in tr.trip_dicts.items():
        if collection_name in activity_dict['collections']:
            activity_ids.append(activity_id)
    
    activities = {}
    for activity_id in activity_ids:
        activity_dict = import_activity(activity_id, client,reload=reload)
        activity_dicts[activity_id] = activity_dict
        start_lat, start_lon = activity_dict['start_latlng']
        end_lat, end_lon = activity_dict['end_latlng']
        activities[activity_id] = (start_lat, start_lon, end_lat, end_lon)

    sorted_activities = []; sorted_activities_reversed = []
    last_end_coords = start_coords
    # Sort activities based on proximity to the end point of the last activity
    while len(sorted_activities) < len(activity_ids):
        closest_activity = None
        min_distance = float('inf')
        activity_reversed = False
        for activity_id,_ in activities.items():
            if activity_id in sorted_activities:
                continue
            for _reversed in [True, False]:
                if _reversed:
                    curr_coords = (activities[activity_id][2], activities[activity_id][3])
                else:
                    curr_coords = (activities[activity_id][0], activities[activity_id][1])
                
                distance = geodesic(last_end_coords, curr_coords).kilometers
                if distance < min_distance:
                    min_distance = distance
                    closest_activity = activity_id
                    activity_reversed = _reversed
        sorted_activities.append(closest_activity)
        activity_dicts[closest_activity]['reversed'] = activity_reversed
        sorted_activities_reversed.append(activity_reversed)
        if activity_reversed:
            last_end_coords = (activities[closest_activity][0], activities[closest_activity][1])
        else:
            last_end_coords = (activities[closest_activity][2], activities[closest_activity][3])
    
    activity_arr = []
    if sort_by == 'location':
        for activity_id in sorted_activities:
            activity_arr.append(activity_dicts[activity_id])
    elif sort_by == 'datetime':
        #activity_arr = sorted(activity_dicts, key=lambda x: x['start_date'])
        activity_arr = sorted(activity_dicts.values(), key=lambda x: x['start_date'])
    return activity_arr

def import_activity(activity_id, client, reload = False, reversed = False):
    # Ensure the 'data' directory exists
    os.makedirs('data', exist_ok=True)
    
    # File path for the pickle file
    file_path = f"data/{activity_id}.pkl"
    #
    # Check if the activity file already exists
    loading_error = False
    if os.path.exists(file_path) and reload == False:
        # Load the activity from the pickle file
        #print('Load',file_path)
        try:
            with open(file_path, 'rb') as f:
                activity_dict = pickle.load(f)
        except:
            loading_error = True
        
    if loading_error or reload or not os.path.exists(file_path):
        #print('Fetching')
        # Fetch the activity from Strava
        activity = client.get_activity(activity_id)

        stream_types = ['latlng', 'altitude', 'distance', 'velocity_smooth', 'heartrate', 'cadence', 'temp', 'time']
        streams = client.get_activity_streams(activity_id, types=stream_types)
        stream_dict = {}
        for stream_type in stream_types:
            stream = streams.get(stream_type)
    
            # Extract the data from the Stream object or use an empty list if not present
            if stream:
                stream_dict[stream_type] = np.array(stream.data)
            else:
                stream_dict[stream_type] = np.array([])
        
        # Convert the activity to a dictionary
        activity_dict = activity.dict()
        activity_dict['stream_dict'] = stream_dict
        activity_dict['activity_photos'] = get_activity_photos_from_web(activity_id)
        # for idx, url in enumerate(activity_dict['activity_photos']):
        #     print(f"Figure {idx+1}: {url}")
        # Save the dictionary to a pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(activity_dict, f)
        print(f"Saved activity {activity_id} to {file_path} ({activity_dict['name']})")    
    
    def get_types(dickt):
        for key,item in dickt.items():
            if isinstance(item, dict):
                get_types(item)
            else:
                print(key, type(item))
    #get_types(activity_dict)
    if reversed:
        for stream_type, stream in activity_dict['stream_dict'].items():
            if stream_type == 'distance':
                activity_dict['stream_dict'][stream_type] = np.amax(stream) - np.flipud(stream)
            else:
                #pass
                activity_dict['stream_dict'][stream_type] = np.flipud(stream) #stream[::-1]
    return activity_dict
