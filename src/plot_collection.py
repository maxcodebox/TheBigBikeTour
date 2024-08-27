#!/usr/bin/env python
# coding: utf-8

# Notes:
# https://github.com/etpinard/plotly-dashboards/tree/master/hover-images
import argparse
from stravalib import Client
import re
import os
import pickle
import trips as tr
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import polyline
import numpy as np
import requests
import os
import tokens as tokens
import numpy as np
from geopy.distance import geodesic
from datetime import datetime
import pytz
from timezonefinder import TimezoneFinder

#
import Polylineparser as pp
import StravaParser as sp


def emoji_to_html(emoji):
    # Convert each character in the flag emoji to its Unicode code point
    code_points = [ord(char) for char in emoji]
    
    # Convert code points to HTML entities
    html_entities = ''.join(f'&#x{code_point:X};' for code_point in code_points)
    
    return html_entities

def zoom_center(
    lons: tuple = None,
    lats: tuple = None,
    lonlats: tuple = None,
    format: str = "lonlat",
    projection: str = "mercator",
    width_to_height: float = 2.0,
) -> (float, dict):
    """Finds optimal zoom and centering for a plotly mapbox.
    Must be passed (lons & lats) or lonlats.
    Temporary solution awaiting official implementation, see:
    https://github.com/plotly/plotly.js/issues/3434

    Parameters
    --------
    lons: tuple, optional, longitude component of each location
    lats: tuple, optional, latitude component of each location
    lonlats: tuple, optional, gps locations
    format: str, specifying the order of longitud and latitude dimensions,
        expected values: 'lonlat' or 'latlon', only used if passed lonlats
    projection: str, only accepting 'mercator' at the moment,
        raises `NotImplementedError` if other is passed
    width_to_height: float, expected ratio of final graph's with to height,
        used to select the constrained axis.

    Returns
    --------
    zoom: float, from 1 to 20
    center: dict, gps position with 'lon' and 'lat' keys

    >>> print(zoom_center((-109.031387, -103.385460),
    ...     (25.587101, 31.784620)))
    (5.75, {'lon': -106.208423, 'lat': 28.685861})
    """
    if lons is None and lats is None:
        if isinstance(lonlats, tuple):
            lons, lats = zip(*lonlats)
        else:
            raise ValueError("Must pass lons & lats or lonlats")

    maxlon, minlon = max(lons), min(lons)
    maxlat, minlat = max(lats), min(lats)
    center = {
        "lon": round((maxlon + minlon) / 2, 6),
        "lat": round((maxlat + minlat) / 2, 6),
    }

    # longitudinal range by zoom level (20 to 1)
    # in degrees, if centered at equator
    lon_zoom_range = np.array(
        [
            0.0007,
            0.0014,
            0.003,
            0.006,
            0.012,
            0.024,
            0.048,
            0.096,
            0.192,
            0.3712,
            0.768,
            1.536,
            3.072,
            6.144,
            11.8784,
            23.7568,
            47.5136,
            98.304,
            190.0544,
            360.0,
        ]
    )

    if projection == "mercator":
        margin = 1.2
        height = (maxlat - minlat) * margin * width_to_height
        width = (maxlon - minlon) * margin
        lon_zoom = np.interp(width, lon_zoom_range, range(20, 0, -1))
        lat_zoom = np.interp(height, lon_zoom_range, range(20, 0, -1))
        zoom = round(min(lon_zoom, lat_zoom), 2)
    else:
        raise NotImplementedError(f"{projection} projection is not implemented")

    return zoom, center


def insert_line_breaks(text, max_length=80):
    """
    Inserts <br> tags into the text to ensure each line is at most max_length characters long.

    Parameters:
    - text: The input string to be formatted.
    - max_length: Maximum length of each line before inserting a <br> tag.

    Returns:
    - A string with <br> tags inserted to format lines according to max_length.
    """
    lines = []
    while len(text) > max_length:
        # Find the last space within the max_length limit
        split_point = text.rfind(" ", 0, max_length)

        # If no space is found, split at max_length
        if split_point == -1:
            split_point = max_length

        # Append the line to the result and remove it from the text
        lines.append(text[:split_point])
        text = text[split_point:].lstrip()

    # Append the remaining part of the text
    lines.append(text)

    # Join all lines with <br> tags
    return "<br>".join(lines)


def get_hoverdata(activity_dict, reversed=False):
    # Create the hover template
    hovertemplate = (
        "<b>Activity Name:</b> %{customdata[0]}<br>"
        "<b>Description:</b> %{customdata[1]}<br>"
        "<b>Distance:</b> %{customdata[2]} km<br>"
        "<b>Elevation Gain:</b> %{customdata[3]} m<br>"
        "<b>Reversed direction:</b> %{customdata[4]}<br>"
        # "<img src='%{customdata[4]}' width='600' height='400'>"
        "<extra></extra>"  # This removes the trace name from the hover text
    )
    customdata = [
        [
            emoji_to_html(activity_dict["name"]),  # Activity name
            insert_line_breaks(
                activity_dict["description"].replace("\n", "<br>"), max_length=80
            ),  # Description
            round(activity_dict["distance"] * 1e-3, 1),  # Distance in km
            activity_dict["total_elevation_gain"],  # Elevation gain in meters
            reversed,
            # photo_url,                                                          # Photo URL
        ]
    ]
    return hovertemplate, customdata


def add_activity_line_to_map(
    fig,
    activity_dict,
    line_color="blue",
    line_width=4.5,
    row=None,
    col=None,
    reversed=False,
):
    # Decode the polyline coordinates
    coordinates = polyline.decode(activity_dict["map"]["summary_polyline"])
    lats, lons = zip(*coordinates)

    # Check for photo URL if available
    photo_url = ""
    if activity_dict["photos"]["count"] > 0:
        photo_url = activity_dict["photos"]["primary"]["urls"]["600"]
    hovertemplate, customdata = get_hoverdata(activity_dict, reversed=reversed)
    # Add the activity line to the map with hover information
    trace = go.Scattermapbox(
        mode="lines",
        lon=lons,
        lat=lats,
        name=activity_dict["name"],
        customdata=customdata
        * len(lons),  # Ensure customdata has the same length as lons
        hovertemplate=hovertemplate,
        line=dict(width=line_width, color=line_color),
        showlegend=False,
    )

    if row is not None and col is not None:
        fig.add_trace(trace, row=row, col=col)
    else:
        fig.add_trace(trace)

    #fig.update_traces(hovertemplate=hovertemplate)
    return fig


# Initialize TimezoneFinder
tz_finder = TimezoneFinder()


def utc_to_local(utc_time, longitude, latitude):
    if longitude is None or latitude is None:
        raise ValueError("Longitude and latitude must not be None")

    if not (-180.0 <= longitude <= 180.0) or not (-90.0 <= latitude <= 90.0):
        raise ValueError("Longitude and latitude must be within valid ranges")

    # Get the time zone based on latitude and longitude
    time_zone_str = tz_finder.timezone_at(lat=latitude, lng=longitude)
    if time_zone_str is None:
        raise ValueError("Could not determine the time zone for the given coordinates")

    # Create timezone object
    local_tz = pytz.timezone(time_zone_str)

    # Check if the datetime is timezone-aware
    if utc_time.tzinfo is None:
        # If naive, localize to UTC
        utc_time = pytz.utc.localize(utc_time)

    # Convert UTC time to local time
    local_time = utc_time.astimezone(local_tz)

    return local_time


def save_collection_summary(collection_name, client, reload=False):
    # Generate the map subplot
    activity_ids, activity_reversed = sp.get_activity_ids(
        collection_name, sort=True, reload=reload, client=client
    )

    collection_summary = {
        "distance_km": 0,
        "moving_time_h": 0,
        "elevation_gain_m": 0,
        "dates": set([]),
        "days": 0,
        "countries": 0,
        "name": tr.collection_dict[collection_name]["name"],
        "flags": set(),
    }

    for idx, (activity_id, reversed) in enumerate(zip(activity_ids, activity_reversed)):

        activity_dict = sp.import_activity(activity_id, client, reload=False)
        name = activity_dict["name"]
        def extract_flag_emojis(text):
            # Regular expression to match pairs of regional indicator symbols
            flag_pattern = re.compile(r'[\U0001F1E6-\U0001F1FF]{2}')
            flags = flag_pattern.findall(text)
            if '🇪🇺' in flags:
                flags.remove('🇪🇺')
            return flags
        for flag in extract_flag_emojis(name):
            collection_summary['flags'].add(flag)
        # date = datetime.fromisoformat(str(activity_dict['start_date'])).strftime("%Y-%m-%d")
        utc_time = datetime.fromisoformat(str(activity_dict["start_date"]))
        longitude = activity_dict["stream_dict"]["latlng"][0][1]
        latitude = activity_dict["stream_dict"]["latlng"][0][0]
        local_time = utc_to_local(utc_time, longitude, latitude)
        collection_summary["dates"].add(local_time.strftime("%Y-%m-%d"))
        collection_summary["days"] += 1
        collection_summary["distance_km"] += activity_dict["distance"] * 1e-3
        collection_summary["moving_time_h"] += activity_dict["moving_time"] / (60 * 60)
        collection_summary["elevation_gain_m"] += activity_dict["total_elevation_gain"]
    #print(´)
    collection_summary["days"] = len(collection_summary["dates"])
    with open(f"data/{collection_name}_summary.pkl", "wb") as f:
        pickle.dump(collection_summary, f)


def plot_collection_combined(collection_name, client, reload=False):
    # Create subplots: 3 rows, 1 column
    ROW_ALTITUDE = 1
    ROW_MAP = 2
    ROW_TABLE = 3
    total_height = 1800  # Increased height to accommodate the table subplot

    fig = make_subplots(
        rows=3,
        cols=1,
        row_heights=[0.2, 0.7, 1.1],
        vertical_spacing=0.03,
        specs=[[{"type": "scatter"}], [{"type": "scattermapbox"}], [{"type": "table"}]],
    )

    # Generate the map subplot
    activity_ids, activity_reversed = sp.get_activity_ids(
        collection_name, sort=True, reload=reload, client=client
    )

    all_lats = []
    all_lons = []

    # Generate a consistent color for each activity
    colors = [
        f"rgb({r},{g},{b})"
        for r, g, b in np.random.randint(0, 255, (len(activity_ids), 3))
    ]

    for idx, (activity_id, reversed) in enumerate(zip(activity_ids, activity_reversed)):
        activity_dict = sp.import_activity(activity_id, client, reload=False)
        coordinates = polyline.decode(activity_dict["map"]["polyline"])
        lats, lons = zip(*coordinates)
        all_lats.extend(lats)
        all_lons.extend(lons)

        # Add the map line with the assigned color
        fig = add_activity_line_to_map(
            fig,
            activity_dict,
            row=ROW_MAP,
            col=1,
            line_color=colors[idx],
            reversed=reversed,
        )

    zoom, center = zoom_center(lons=all_lons, lats=all_lats, width_to_height=5.0)

    fig.update_layout(
        mapbox={
            "accesstoken": tokens.MAPBOX_TOKEN,
            "style": "outdoors",
            "center": center,
            "zoom": zoom,
        }
    )

    # Generate the altitude subplot
    N = 10
    x0 = 0
    for idx, (activity_id, reversed) in enumerate(zip(activity_ids, activity_reversed)):
        activity_dict = sp.import_activity(
            activity_id, client, reload=reload, reversed=reversed
        )
        hovertemplate, customdata = get_hoverdata(activity_dict, reversed=reversed)

        # Use the same color for the altitude plot as in the map
        fig.add_trace(
            go.Scatter(
                x=(activity_dict["stream_dict"]["distance"][::N] + x0) * 1e-3,
                y=activity_dict["stream_dict"]["altitude"][::N],
                mode="lines",
                line=dict(width=4.5, color=colors[idx]),  # Apply the same color
                customdata=customdata
                * len(activity_dict["stream_dict"]["distance"][::N]),
                hovertemplate=hovertemplate,
                name=activity_dict["name"],
                fill="tozeroy",
            ),
            row=ROW_ALTITUDE,
            col=1,
        )
        x0 += np.amax(activity_dict["stream_dict"]["distance"])

    # Create table data
    table_header = [
        "Name",
        "Date",
        # "Time",
        "Distance (km)",
        "Moving Time (h)",
        "Elevation Gain (m)",
        "Max Elevation (m)",
        "Average Heart Rate (bpm)",
        "Max Heart Rate (bpm)",
        "Strava Link",
    ]
    table_cells = []

    for activity_id in activity_ids:
        activity_dict = sp.import_activity(activity_id, client, reload=False)
        distance_km = activity_dict["distance"] * 1e-3
        moving_time_h = activity_dict["moving_time"] / 3600
        elevation_gain_m = activity_dict["total_elevation_gain"]
        max_elevation_m = activity_dict["elev_high"]
        avg_heart_rate_bpm = activity_dict.get("average_heartrate", "N/A")
        max_heart_rate_bpm = activity_dict.get("max_heartrate", "N/A")
        strava_link = f"https://www.strava.com/activities/{activity_id}"
        # Format the datetime object to the desired format
        date = datetime.fromisoformat(str(activity_dict["start_date"])).strftime(
            "%Y-%m-%d"
        )
        time = datetime.fromisoformat(str(activity_dict["start_date"])).strftime(
            "%H:%M"
        )
        # date = activity_dict['start_date'] #datetime.strptime(activity_dict['start_date'], '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d')

        table_cells.append(
            [
                activity_dict["name"],
                date,
                # time,
                f"{distance_km:.2f}",
                f"{moving_time_h:.2f}",
                f"{elevation_gain_m:.2f}",
                f"{max_elevation_m:.2f}",
                avg_heart_rate_bpm,
                max_heart_rate_bpm,
                strava_link,
            ]
        )

    # Add table trace without changing the cell background color
    fig.add_trace(
        go.Table(
            header=dict(
                values=table_header,
                fill_color="paleturquoise",
                align="left",
                font_size=12,
            ),
            cells=dict(
                values=list(zip(*table_cells)),
                fill_color="lavender",
                align="left",
                font_size=11,
            ),
        ),
        row=ROW_TABLE,
        col=1,
    )

    # Adjust layout for the combined figure
    fig.update_layout(
        height=total_height,  # Increase the height to accommodate the table subplot
        margin=dict(l=20, r=20, t=10, b=10),
        showlegend=False,
    )
    fig.update_xaxes(title_text="Distance (km)", row=1, col=1)
    fig.update_yaxes(title_text="Altitude (m)", row=1, col=1)

    os.makedirs("figures/html", exist_ok=True)
    html_path = f"figures/html/{collection_name}.html"
    fig.write_html(html_path)
    print(f"open {html_path}")

    # Isolate the map subplot
    map_fig = go.Figure()

    for trace in fig.data:
        # Check if the trace belongs to the map subplot (row=2, col=1)
        if trace.type == "scattermapbox":
            map_fig.add_trace(trace)

    # Update the layout to match the original map subplot
    map_fig.update_layout(
        mapbox={
            "accesstoken": tokens.MAPBOX_TOKEN,
            "style": "outdoors",
            "center": fig.layout.mapbox["center"],
            "zoom": fig.layout.mapbox["zoom"],
        },
        height=600,  # Adjust height as needed
        margin=dict(l=0, r=0, t=0, b=0),
    )

    # Save the isolated map subplot as PNG
    os.makedirs("figures/static", exist_ok=True)
    png_path = f"figures/static/{collection_name}_map.png"
    map_fig.write_image(png_path)


def update_all():
    client = Client(access_token=tokens.ACCESS_TOKEN)
    activity_ids = [trip for trip, _ in tr.trip_dicts.items()]
    for activity_id in activity_ids:
        print(f"Updating {activity_id}")
        activity_dict = sp.import_activity(activity_id, client, reload=True)


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Run the script with a collection.")

    # Get the possible collections from tr.collection_dict
    possible_collections = [col for col, _ in tr.collection_dict.items()] + ["all"]

    # Add an argument for the collection
    parser.add_argument(
        "collection",
        choices=possible_collections,
        help="Name of the collection to be processed.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Initialize the client
    client = Client(access_token=tokens.ACCESS_TOKEN)

    # Get the collection name from the arguments
    collection = args.collection
    if collection == "all":
        collections = [
            "norway-turkey",
            "berlin-tarifa",
            "hue-hcmc_2016",
            "taiwan_2017",
            "hue-hcmc_2016",
            "yokohama-fukuoka_2019",
            "bavarian-alp-traverse",
        ]
    else:
        collections = [collection]
    for collection in collections:
        plot_collection_combined(collection, client, reload=False)
        #save_collection_summary(collection, client, reload=False)


if __name__ == "__main__":
    main()
    #update_all()
