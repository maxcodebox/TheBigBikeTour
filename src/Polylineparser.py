import numpy as np
from geopy.distance import geodesic
import polyline as pl
import requests

def decode(pline):
    coordinates = pl.decode(pline, min_distance_km)



def get_min_max_lat_lon(pline):
    """
    Get the minimum and maximum latitude and longitude from an activity's polyline.

    Parameters:
    - activity_dict: Dictionary containing the activity data with a 'map' key.

    Returns:
    - min_lat: Minimum latitude value.
    - max_lat: Maximum latitude value.
    - min_lon: Minimum longitude value.
    - max_lon: Maximum longitude value.
    """
    # Decode the polyline to get a list of (latitude, longitude) pairs
    coordinates = pl.decode(pline)

    # Unzip the coordinates into separate lists of latitudes and longitudes
    lats, lons = zip(*coordinates)

    # Find the minimum and maximum latitudes and longitudes
    min_lat = min(lats)
    max_lat = max(lats)
    min_lon = min(lons)
    max_lon = max(lons)

    return min_lat, max_lat, min_lon, max_lon

def get_distance_array_km(pline):
    # Decode the polyline to get a list of (latitude, longitude) pairs
    coordinates = pl.decode(pline)

    # Unzip the coordinates into separate lists of latitudes and longitudes
    lats, lons = zip(*coordinates)

    # Initialize an empty list to store distances
    distances = []

    # Iterate over consecutive points to calculate distances
    for i in range(1, len(coordinates)):
        point1 = coordinates[i-1]
        point2 = coordinates[i]
        
        # Calculate the geodesic distance between consecutive points
        distance = geodesic(point1, point2).kilometers  # or .kilometers, .miles etc.
        distances.append(distance)

    # Optional: Compute the cumulative distance array
    cumulative_distances = np.cumsum(distances)

    return cumulative_distances

def get_altitude_array_meters(pline):
    # Decode the polyline to get a list of (latitude, longitude) pairs
    coordinates = pl.decode(pline)

    # Prepare the API request
    locations = "|".join(f"{lat},{lon}" for lat, lon in coordinates)
    print(locations)
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations}"
    print(url)

    exit()
    # Send the request to the Open-Elevation API
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
    else:
        raise Exception(f"Error fetching elevation data: {response.status_code}")

    # Extract altitude values from the response
    altitudes = [result['elevation'] for result in data['results']]

    # Initialize an empty list to store altitude differences
    altitude_changes = []

    # Iterate over consecutive points to calculate altitude differences
    for i in range(1, len(altitudes)):
        altitude_diff = abs(altitudes[i] - altitudes[i-1])
        altitude_changes.append(altitude_diff)

    # Compute the cumulative altitude difference array
    cumulative_altitude_changes = np.cumsum(altitude_changes)

    return cumulative_altitude_changes

