#!/usr/bin/env python
# coding: utf-8

# Notes:
# https://github.com/etpinard/plotly-dashboards/tree/master/hover-images
import argparse
import re
import os
import pickle
import pandas as pd
import trips as tr
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.colors as pc
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

def float_hours_to_hhmm(hours_float):
    # Extract the integer part of the hours
    hours = int(hours_float)
    
    # Calculate the minutes from the fractional part
    minutes = int(round((hours_float - hours) * 60))
    
    # Format the hours and minutes to ensure two digits for each
    return f"{hours:02}:{minutes:02}"

def emoji_to_html(emoji):
    # Convert each character in the flag emoji to its Unicode code point
    code_points = [ord(char) for char in emoji]
    
    # Convert code points to HTML entities
    html_entities = ''.join(f'&#x{code_point:X};' for code_point in code_points)
    
    return html_entities
def emoji_to_html_entities(text):
    # Convert each character to its HTML entity if it's an emoji
    return ''.join(
        f'&#x{ord(char):X};' if ord(char) > 0x1F600 else char
        for char in text
    )

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
    #print(activity_dict["description"])
    # print(emoji_to_html_entities(insert_line_breaks(activity_dict["description"], max_length=80)))
    # input_text = "I love 🍕 and 😄!"
    # output_text = emoji_to_html_entities(input_text)
    # print(output_text)
    newline_char = '\n'
    newline_char_html = emoji_to_html(newline_char)
    period_char = '. '
    period_char_html = emoji_to_html(period_char)
    def fix_description(text):
        text = emoji_to_html(text)
        for char in ['\n','. ','! ',', ']:
            char_html = emoji_to_html(char)
            text = text.replace(char_html, f"{char_html}<br>")
        for char in ['⛰️']:
            char_html = emoji_to_html(char)
            text = text.replace(char_html, f"<br>{char_html}")
        return text
    customdata = [
        [
            emoji_to_html(activity_dict["name"]),  # Activity name
            # insert_line_breaks(emoji_to_html(activity_dict["description"]), max_length=80),#.replace("\n", "<br>"),  # Description
            #emoji_to_html(activity_dict["description"]).replace(emoji_to_html('⛰️'),f"<br>{emoji_to_html('⛰️')}").replace(newline_char_html,"<br>").replace(period_char_html,'.<br>'),#.replace("\n", "<br>"),
            fix_description(activity_dict["description"]),
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
    line_width=3.0,
    row=None,
    col=None,
    reversed=False,
):
    # Decode the polyline coordinates
    coordinates = polyline.decode(activity_dict["map"]["summary_polyline"])
    lats, lons = zip(*coordinates)
    elevations = activity_dict["stream_dict"]["altitude"]

    hovertemplate, customdata = get_hoverdata(activity_dict, reversed=reversed)
    # Add the activity line to the map with hover information
    # 2D map
    for i in [0,1]:
        if i == 0:
            lw = line_width + 2
            color = 'white'
        else:
            lw = line_width
            color = line_color
        trace = go.Scattermapbox(
            mode="lines",
            lon=lons,
            lat=lats,
            name=activity_dict["name"],
            customdata=customdata
            * len(lons),  # Ensure customdata has the same length as lons
            hovertemplate=hovertemplate,
            line=dict(width=lw, color=color),
            showlegend=False,
        )

        # Add the activity line to the 3D map with hover information
        # 3D map
        # trace = go.Scatter3d(
        #     mode="lines",
        #     x=lons,  # X-axis: Longitude
        #     y=lats,  # Y-axis: Latitude
        #     z=elevations,  # Z-axis: Elevation or height
        #     name=activity_dict["name"],
        #     customdata=customdata * len(lons),  # Ensure customdata has the same length as lons
        #     hovertemplate=hovertemplate,
        #     line=dict(width=line_width, color=line_color),
        #     showlegend=False,
        # )
            
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

def extract_flag_emojis(text):
    # Regular expression to match pairs of regional indicator symbols
    flag_pattern = re.compile(r'[\U0001F1E6-\U0001F1FF]{2}')
    flags = flag_pattern.findall(text)
    if '🇪🇺' in flags:
        flags.remove('🇪🇺')
    return flags

def plot_elevation_profile(activities):
    for activity_dict in activities:
        
        fig = go.Figure()
        # Extract the altitude and distance data
        altitude = activity_dict["stream_dict"]["altitude"]
        distance = activity_dict["stream_dict"]["distance"] * 1e-3
        
        # Add a trace for this activity
        fig.add_trace(go.Scatter(x=distance, y=altitude, mode='lines', name=activity_dict["name"], line=dict(color='black')))
        
        # Find the minimum and maximum altitudes
        min_alt = np.amin(altitude)
        max_alt = np.amax(altitude)
        
        # Get the indices of the min and max altitude points
        min_idx = np.argmin(altitude)
        max_idx = np.argmax(altitude)
        
        # Get the corresponding distances
        min_dist = distance[min_idx]
        max_dist = distance[max_idx]
        
        # Add dashed lines at min and max altitude
        d0 = distance[0]
        d1 = distance[-1]
        # fig.add_trace(go.Scatter(x=[d0, d1], y=[min_alt, min_alt], mode='lines', name=activity_dict["name"], line=dict(color='gray', dash='dot')))
        # fig.add_trace(go.Scatter(x=[d0, d1], y=[max_alt, max_alt], mode='lines', name=activity_dict["name"], line=dict(color='gray', dash='dot')))
        font=dict(size=24)
        # Add text annotation above the highest point
        fig.add_annotation(
            x=max_dist,
            y=max_alt,
            text=f"{max_alt:.0f}m",
            font=font,
            showarrow=True,
            arrowhead=2,
            arrowwidth=2,
            ax=0,
            ay=-40  # Offset the text above the point
        )
        
        # Add text annotation below the lowest point
        fig.add_annotation(
            x=min_dist,
            y=min_alt,
            text=f"{min_alt:.0f} m",
            font=font,
            showarrow=True,
            arrowhead=2,
            arrowwidth=2,
            ax=0,
            ay=40  # Offset the text below the point
        )
        
        # Update layout to remove axis labels and background color
        fig.update_layout(
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
            showlegend=False,  # Remove the legend
            margin=dict(l=0, r=0, t=0, b=40),  # Set all margins to zero
        )

        # Save the plot as a small PNG file
        fig.write_image(f"figures/static/{activity_dict['id']}_elevation_profile.png", format="png", width=800, height=400)

def save_collection_summary(collection_name,activities):

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
    for activity_dict in activities:
        name = activity_dict["name"]
        
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
        collection_summary['primary_photo_url'] = ''
        
    collection_summary["days"] = len(collection_summary["dates"])
    with open(f"data/{collection_name}_summary.pkl", "wb") as f:
        pickle.dump(collection_summary, f)


def plot_collection_combined(collection_name, activities):
    # Create subplots: 3 rows, 1 column
    ROW_ALTITUDE = 1
    ROW_MAP = 2
    #ROW_TABLE = 3
    total_height = 900  # Increased height to accommodate the table subplot

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.25, 0.75],
        vertical_spacing=0.05,
        specs=[
                [{"type": "scatter"}],
                [{"type": "scattermapbox"}],
                # [{"type": "scatter3d"}],
                # [{"type": "table"}]
               ],
    )

    all_lats = []
    all_lons = []

    # Generate a consistent color for each activity
    # colors = [
    #     f"rgb({r},{g},{b})"
    #     for r, g, b in np.random.randint(0, 255, (len(activities), 3))
    # ]
    # colors = [
    #     f"rgb({r},{g},{b})"
    #     for r, g, b in np.random.randint(0, 200, (len(activities), 3))  # Range set to 0-100 for darker colors
    # ]

    # Get the colors from the Viridis colorscale
    colors_inferno = pc.sample_colorscale('Inferno', [i / (len(activities) - 1) for i in range(len(activities))])
    colors_dark24 = px.colors.qualitative.Dark24
    colors = [colors_dark24[i % len(colors_dark24)] for i in range(len(activities))]
    #print(colors)
    # print(f"{viridis_colors = }")
    # exit()
    for idx,activity_dict in enumerate(activities):
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
            reversed=activity_dict['reversed'],
        )

    zoom, center = zoom_center(lons=all_lons, lats=all_lats, width_to_height=5.0)

    

    # Generate the altitude subplot
    N = 50
    x0 = 0

    for idx,activity_dict in enumerate(activities):
        hovertemplate, customdata = get_hoverdata(activity_dict, reversed=activity_dict['reversed'])

        if activity_dict['reversed']:
            x = activity_dict["stream_dict"]["distance"][-1] - activity_dict["stream_dict"]["distance"][::-1]
            y = activity_dict["stream_dict"]["altitude"][::-1]
        else:
            x = activity_dict["stream_dict"]["distance"]
            y = activity_dict["stream_dict"]["altitude"]
        # Use the same color for the altitude plot as in the map
        fig.add_trace(
            go.Scatter(
                x=(x[::N] + x0) * 1e-3,
                y=y[::N],
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
    
    # Update the layout
    
    # Define all available Mapbox styles
    styles = [
        "outdoors",
        "streets",
        "light",
        "dark",
        "satellite",
        "satellite-streets",
        # "navigation-day",
        # "navigation-night"
    ]
    # Create buttons for each style
    buttons = [
        dict(
            label=style.capitalize().replace("-", " "),
            method="relayout",
            args=["mapbox.style", style]
        ) for style in styles
    ]
    # Set up the layout with default "streets" style
    fig.update_layout(
        mapbox={
            "accesstoken": tokens.MAPBOX_TOKEN,
            "style": "outdoors",
            # "style":'satellite',
            "center": center,
            "zoom": zoom,
        },
        updatemenus=[
            dict(
                type="dropdown",
                x=0.05,
                y=0.70,
                buttons=buttons,
                showactive=True,
                xanchor='left',
                yanchor='top'
            )
        ],
        height=total_height,  # Increase the height to accommodate the table subplot
        margin=dict(l=20, r=20, t=10, b=10),
        showlegend=False,
    )
    fig.update_xaxes(title_text="Distance (km)", row=1, col=1)
    fig.update_yaxes(title_text="Altitude (m)", row=1, col=1)
    #fig.show()
    
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
    
    # map_fig.update_layout(mapbox_style="open-street-map")
    # Save the isolated map subplot as PNG
    os.makedirs("figures/static", exist_ok=True)
    png_path = f"figures/static/{collection_name}_map.png"
    map_fig.write_image(png_path)

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
    for activity_dict in activities:
        activity_id = activity_dict['id']
        distance_km = activity_dict["distance"] * 1e-3
        moving_time_h = activity_dict["moving_time"] / 3600
        elevation_gain_m = activity_dict["total_elevation_gain"]
        max_elevation_m = activity_dict["elev_high"]
        avg_heart_rate_bpm = activity_dict.get("average_heartrate", "N/A")
        avg_heart_rate_bpm_str = f"{avg_heart_rate_bpm:.0f}" if avg_heart_rate_bpm is not None else "No data"
        max_heart_rate_bpm = activity_dict.get("max_heartrate", "N/A")
        max_heart_rate_bpm_str = f"{max_heart_rate_bpm:.0f}" if max_heart_rate_bpm is not None else "No data"
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
                # f'<a href="{strava_link}">{activity_dict["name"]}</a>',
                date,
                # time,
                f"{distance_km:.1f}",
                f"{moving_time_h:.1f}",
                f"{elevation_gain_m:.0f}",
                f"{max_elevation_m:.0f}",
                f"{avg_heart_rate_bpm_str}",
                f"{max_heart_rate_bpm_str}",
                strava_link,
            ]
        )
    
    df = pd.DataFrame(columns=table_header, data=table_cells)
    # Convert 'Date' column to datetime format (if not already in datetime format)
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort the DataFrame by the 'Date' column in ascending order
    df = df.sort_values(by='Date')

    # Define a function to format link with custom text
    def format_link(url):
        return f'<a href="{url}" target="_blank">View Activity</a>'

    # Apply formatting to the Strava Link column
    df['Strava Link'] = df['Strava Link'].apply(format_link)
    # Convert DataFrame to HTML
    # df.style.set_table_styles([{'selector': 'table', 'props': [('max-width', '600px')]}]).render()
    html_table = df.to_html(escape=False, index=False)
    html_table = f"""
    <div style="max-width: 100%; overflow-x: auto;">
        {html_table}
    </div>
    """
    # Define CSS styles
    css_styles = """
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f4f4f4;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 0px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #04AA6D;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        tr:hover {
            background-color: #f1f1f1;
        }
        a {
            color: #007bff;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
    """
    # Combine CSS and HTML table
    html_output = f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>{collection_name}</title>{css_styles}</head><body>{html_table}</body></html>'

    # Write to an HTML file
    os.makedirs("figures/html", exist_ok=True)
    with open(f'figures/html/{collection_name}_summary_table.html', 'w') as f:
        f.write(html_output)
    print(f'open figures/html/{collection_name}_summary_table.html')
    # for activity_dict in activities:
    #     pattern = r'⛰️\s*([^\(]+)\s*\(([\d,]+)\s*m\)'
    #     matches = re.findall(pattern, activity_dict['description'])
    #     # Process and print the matches
    #     for peak, elevation in matches:
    #         # Replace comma with dot and convert to float
    #         elevation_float = float(elevation.replace(',', '.'))

    # Adjust layout for the combined figure
    

def plot_photogallery(collection_name, activities):
    setup_gallery_html = ""
    gallery_html = ""
    for iact,activity_dict in enumerate(activities):
        strava_link = f"https://www.strava.com/activities/{activity_dict['id']}"
        
        additional_photos_html = ""; first_photo = ""; gallery_part = ""
        if activity_dict['photos']['count'] > 0:
            for iphoto,photo in enumerate(activity_dict['activity_photos']):
                if iphoto == 0:
                    first_photo = photo['urls']['5000']
                additional_photos_html += f"""<img src="{photo['urls']['5000']}" class="thumbnail" alt="Thumbnail {iphoto}">"""
            gallery_part = f"""
            <div class="photo-gallery" id="gallery{iact}">
                    <!-- Large Photo Display -->
                    <div class="large-photo-container">
                        <img class="large-photo" src="{first_photo}" alt="Large Photo 2">
                    </div>
                    <!-- Thumbnail Row -->
                    <div class="thumbnail-row">
                        {additional_photos_html}
                    </div>
                </div>
            """
        
        gallery_html += f"""
        <div id="gallery-container">
            <div class="gallery-title">
                <h1>
                    {activity_dict['name']}
                    <a href="{strava_link}" target="_blank">
                        <img src="figures/strava.svg" alt="Strava logo" class="logo">
                    </a>
                </h1>
                <p>{activity_dict['description']}</p>
                <div class="info-container">
                    <img src="figures/distance-icon.png" alt="Distance icon" class="logo">
                    <span class="info-text">{activity_dict['distance']*1e-3:.0f} km</span>
                    <img src="figures/elevation-up-icon.png" alt="Elevation icon" class="logo">
                    <span class="info-text">{activity_dict['total_elevation_gain']:.0f} hm</span>
                    <img src="figures/time-icon.png" alt="Time icon" class="logo">
                    <span class="info-text">{float_hours_to_hhmm(activity_dict['moving_time'] / (60 * 60))}</span>
                </div>
                <div class="elevation-photo-container">
                    <img class="elevation-photo" src="figures/static/{activity_dict['id']}_elevation_profile.png">
                </div>
                {gallery_part}
            </div>
        </div>
        """
        if activity_dict['photos']['count'] > 0:
            setup_gallery_html += f"setupGallery('gallery{iact}');\n"
    with open("templates/PHOTOGALLERY_TEMPLATE.html", "r") as f:
        template = ''.join(f.readlines())
    #template = template.replace('<!--LARGE_PHOTO-->',first_photo)
    template = template.replace('<!--gallery_content-->',gallery_html)
    template = template.replace('SETUP_GALLERY_CODE',setup_gallery_html)
    # with open(f'figures/html/photogallery_{collection_name}.html','w') as f2:
    with open(f'figures/html/photogallery_{collection_name}.html','w') as f2:
        f2.write(template)
    print(f'open figures/html/photogallery_{collection_name}.html')




def max_average_elevation_gain_in_window(elevation_gain, time_data, window_size):
    """
    Calculate the maximum average elevation gain within a sliding window of size `window_size`.
    """
    max_avg_gain = 0
    window_samples = int(window_size)  # Number of samples corresponding to the time window

    for i in range(len(time_data) - window_samples):
        # Get the current window data
        window_altitude_gain = elevation_gain[i:i + window_samples]
        window_time = time_data[i + window_samples] - time_data[i]  # Time difference over the window

        # Calculate average elevation gain per second within the window
        if window_time > 0:  # Ensure no division by zero
            avg_gain = np.sum(window_altitude_gain) / window_time
            max_avg_gain = max(max_avg_gain, avg_gain)

    return max_avg_gain

def plot_elevation_gain_curve(collection_name, activities, downsample_factor=10):
    # time_windows = [2**n for n in range(4, 14)]  # 16s, 32s, 64s, ..., 1024s
    time_windows = [30,40,60,60*2,60*5,60*10,60*20,60*30,60*60,60*60*2]
    maximum_elevation_gain = [0] * len(time_windows)
    traces = []
    
    for iact, activity_dict in enumerate(activities):
        print(f"Processing activity {iact + 1}/{len(activities)}: {activity_dict['name']}")
        # if not 'time' in activity_dict['stream_dict'].keys():
        #     print(f'No time data for {activity_dict["name"]}, importing from strava...')
        #     activity_dict = sp.import_activity(activity_dict['id'],reload=True)
        # Extract and downsample the required streams
        time_data = activity_dict['stream_dict']['time'][::downsample_factor]  # Assuming time is in seconds
        altitude_data = activity_dict['stream_dict']['altitude'][::downsample_factor]
        
        # Calculate elevation gain between consecutive points
        elevation_gain = np.diff(altitude_data)
        elevation_gain[elevation_gain < 0] = 0  # Discard negative gains (descents)

        # Compute maximum average elevation gain for each time window
        max_avg_gains = []
        for iwindow, window_size in enumerate(time_windows):
            # Adjust window size for downsampled data
            adjusted_window_size = window_size // downsample_factor
            max_avg_gain = max_average_elevation_gain_in_window(elevation_gain, time_data, adjusted_window_size)
            max_avg_gains.append(max_avg_gain)
            maximum_elevation_gain[iwindow] = max(maximum_elevation_gain[iwindow], max_avg_gain)
        
        # Prepare trace for this activity
        traces.append(go.Scatter(
            x=np.array(time_windows) / 60,  # Convert seconds to minutes for the x-axis
            y=np.array(max_avg_gains) * 3600,  # Convert to meters per hour for y-axis
            mode='lines+markers',
            line=dict(width=0.5), opacity=0.5,
            name=activity_dict['name'],
        ))

    # Add the maximum elevation gain trace
    traces.append(go.Scatter(
        x=np.array(time_windows) / 60,  # Convert seconds to minutes for the x-axis
        y=np.array(maximum_elevation_gain) * 3600,  # Convert to meters per hour for y-axis
        mode='lines+markers+text',
        line=dict(color='red', dash='dash', width=2),
        text=[f"{hm:.0f} m/h" for hm in np.array(maximum_elevation_gain) * 3600],  # Display y-values as text
        textposition='top right',  # Position text above the right side of each marker
        name='Maximum'
    ))

    # Create the layout
    # layout = go.Layout(
    #     # title=f'Elevation Gain Curve for {collection_name}',
    #     xaxis=dict(title='Time (min)', type='log'),
    #     yaxis=dict(title='Maximum elevation gain (m/h)'),
    #     # legend=dict(x=0.1, y=0.9),
    #     hovermode='closest'
    # )
    # Create the layout
    layout = go.Layout(
        # title=f'Elevation Gain Curve for {collection_name}',
        xaxis=dict(
            title='Time (min)',
            type='log',
            tickmode='linear',
            ticks='outside',
            linecolor='black',
            gridcolor='lightgrey',
            showline=True,
            showgrid=True,
            zeroline=True,
            showticklabels=True,
            # tickangle=45,
            tickprefix='',
            tickvals=np.array(time_windows) / 60,  # Ensure all time windows are shown
            ticktext=[f"{int(t)} min" for t in np.array(time_windows) / 60]  # Custom tick labels
        ),
        yaxis=dict(
            title='Maximum elevation gain (m/h)',
            ticks='outside',
            linecolor='black',
            gridcolor='lightgrey',
            showline=True,
            showgrid=True,
            zeroline=True,
            showticklabels=True,
            tickprefix='',
            tickformat=',',  # Format tick labels with thousands separators
            showexponent='none'
        ),
        plot_bgcolor='white',  # Background color of the plot area
        paper_bgcolor='white',  # Background color of the entire figure
        margin=dict(l=50, r=50, t=50, b=50),  # Adjust margins if needed
        # legend=dict(x=0.1, y=0.9),
        hovermode='closest'
    )

    # Create the figure and plot
    fig = go.Figure(data=traces, layout=layout)
    #fig.show()
    # Update layout to remove axis labels and background color
    fig.update_layout(
        # xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        # yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        # plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
        # paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
        # showlegend=False,  # Remove the legend
        margin=dict(l=0, r=0, t=0, b=0),  # Set all margins to zero
    )

    # Save the plot as a small PNG file
    path = f"figures/html/{collection_name}_elevation_gain_curve"
    # fig.write_image(path + '.png', format="png", width=1600, height=800)
    fig.write_html(path + '.html')
    print(f"open {path}.html")

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

    # Get the collection name from the arguments
    collection = args.collection
    if collection == "all":
        collections = [
            # "norway-turkey",
            # "berlin-tarifa",
            "hue-hcmc_2016",
            "taiwan_2017",
            "yokohama-fukuoka_2019",
            "bavarian-alp-traverse",
            "perla-hikes",
            "ibiza_2023",
        ]
    else:
        collections = [collection]
    for collection in collections:
        activities = sp.import_collection(collection, reload=False)
        plot_elevation_profile(activities)
        plot_collection_combined(collection, activities)
        plot_photogallery(collection, activities)
        save_collection_summary(collection, activities)
        plot_elevation_gain_curve(collection, activities)
if __name__ == "__main__":
    main()
    #update_all()
