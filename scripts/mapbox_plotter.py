
import sys
import gpxpy
import glob
import os
from bs4 import BeautifulSoup
#from pandas import DataFrame
import numpy as np
import pandas as pd
import mplleaflet
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import pickle
from pathlib import Path
import matplotlib.patches as patches
import geopy.distance

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from PIL import Image

import countries  # https://github.com/che0/countries
import flagpy as fp
from datetime import datetime
import math
import re


from fit2gpx import StravaConverter
from fit2gpx import Converter

# My packages
import colors as cols
import gpxreader as gpxr
import trips as tr
import texthandler as th
import api_keys as apik
# as per recommendation from @freylis, compile once only
CLEANR = re.compile('<.*?>')

def cleanhtml(raw_html):
  cleantext = re.sub(CLEANR, '', raw_html)
  return cleantext


# pip install --global-option=build_ext --global-option="-I/usr/include/gdal" GDAL==`gdal-config --version`
# http://thematicmapping.org/downloads/world_borders.php
#cc = countries.CountryChecker('world_borders/TM_WORLD_BORDERS_SIMPL-0.3.shp')
cc = countries.CountryChecker('world_borders/TM_WORLD_BORDERS-0.3.shp')


def get_topnav(collections,active_collection = ''):
    topnav_str = '<div class="topnav">\n'
    for collection in ['index'] + collections:
        trip_title = collection
        trip_label = trip_title

        if collection == 'index':
            trip_title = 'Home'
        else:
            trip_title = tr.collection_dict[collection]["name"]
        if trip_label == active_collection:
            topnav_str += f'<a class="active" href="{trip_label}.html">{trip_title}</a>\n'
        else:
            topnav_str += f'<a href="{trip_label}.html">{trip_title}</a>\n'
    topnav_str += '</div>\n'
    return topnav_str


def get_trip_content(collection):
    trip_title = collection
    trip_label = trip_title
    trip_filename = f'trips/{trip_label}.html'
    with open(trip_filename,'r') as f:
        return ''.join(f.readlines())

def zoom_center(lons: tuple=None, lats: tuple=None, lonlats: tuple=None,
        format: str='lonlat', projection: str='mercator',
        width_to_height: float=2.0) -> (float, dict):
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
            raise ValueError(
                'Must pass lons & lats or lonlats'
            )

    maxlon, minlon = max(lons), min(lons)
    maxlat, minlat = max(lats), min(lats)
    center = {
        'lon': round((maxlon + minlon) / 2, 6),
        'lat': round((maxlat + minlat) / 2, 6)
    }

    # longitudinal range by zoom level (20 to 1)
    # in degrees, if centered at equator
    lon_zoom_range = np.array([
        0.0007, 0.0014, 0.003, 0.006, 0.012, 0.024, 0.048, 0.096,
        0.192, 0.3712, 0.768, 1.536, 3.072, 6.144, 11.8784, 23.7568,
        47.5136, 98.304, 190.0544, 360.0
    ])

    if projection == 'mercator':
        margin = 1.2
        height = (maxlat - minlat) * margin * width_to_height
        width = (maxlon - minlon) * margin
        lon_zoom = np.interp(width , lon_zoom_range, range(20, 0, -1))
        lat_zoom = np.interp(height, lon_zoom_range, range(20, 0, -1))
        zoom = round(min(lon_zoom, lat_zoom), 2)
    else:
        raise NotImplementedError(
            f'{projection} projection is not implemented'
        )

    return zoom, center
def plot_collection(
        collection = 'hamburg-paris_2022',
        lat=52,
        lon=1,
        zoom=3.5,
        bool_update_dfs = True,
        bool_update_csv = False,
        bool_update_pkl = False,
        bool_plot_pauses = False,
        bool_update_triphtml = False,
        bool_update_images   = False,
        bool_auto_zoom = True,
        ):




    trip_title              = collection
    trip_label              = trip_title
    index_filename          = f'trips/{trip_label}.html'
    dfcollection_filename   = f'trips/{trip_label}_data.pkl'

    #print(collection)
    if bool_update_triphtml or not os.path.isfile(index_filename):

        # def check_id(activity_id,selected_collections):
        #     for collection in selected_collections:
        #         if collection in tr.trip_dicts[activity_id]['collections']:
        #             return True
        #     return False
        def check_id(activity_id,collection):
            if collection in tr.trip_dicts[activity_id]['collections']:
                return True
            return False


        activity_ids = [activity_id for activity_id in tr.trip_dicts if check_id(activity_id,collection)]
        if bool_update_dfs or not os.path.isfile(dfcollection_filename):

            df_strava = pd.read_csv(apik.DIR_STRAVA + '/activities.csv',
                                    delimiter=',')  # ,quotechar='"')
            df_strava.columns = [c.lower().replace(' ','_').replace('.','') for c in df_strava.columns]

            activity_dict = dict()
            for activity_id in activity_ids:
                activity = df_strava[df_strava['activity_id'] == activity_id].iloc[0].to_dict()

                activity['filepath'] = activity['filename']
                activity['filepath'] = f"{apik.DIR_STRAVA}/{activity['filepath'].replace('.fit.gz','.fit')}"

                #activity['filepath'] = f"{apik.DIR_STRAVA}/{activity['filepath'].replace('activities/','activities_convert/').replace('.fit.gz','.gpx')}"
                activity_dict[activity_id] = activity
            filepaths = [activity['filepath'] for _, activity in activity_dict.items()]
            df_list = [gpxr.read_gpx(activity['filepath'].replace('.gpx.gz','.gpx'), bool_update_csv=bool_update_csv, bool_update_pkl=bool_update_pkl,
                                     activity_id=activity_id) for activity_id, activity in activity_dict.items()]

            trip_collections = set()
            for df in df_list:

                df.strava_link = f"https://www.strava.com/activities/{df.activity_id}"
                df.strava_link_embed = f"<a href=\"{df.strava_link}\">{activity_dict[df.activity_id]['activity_name']}</a>"

                df.veloviewer_link = f"https://veloviewer.com/athletes/18995448/activities/{df.activity_id}"
                df.veloviewer_link_embed = f"<a href=\"{df.veloviewer_link}\">{activity_dict[df.activity_id]['activity_name']}</a>"

                df.link_embeds = f"<a href=\"{df.strava_link}\">Strava</a>, <a href=\"{df.veloviewer_link}\">Veloviewer</a>"


                for key in activity_dict[df.activity_id]:
                    df[key] = activity_dict[df.activity_id][key]

                a = True


                for collection in tr.trip_dicts[df.activity_id]['collections']:

                    if collection in tr.individual_trips:
                        trip_collections.add(collection)
                        df.collection = collection
                        df['collection'] = collection
                        a = False
                        break
                if a:
                    print('NO INDIVIDUAL COLLECTION NAME! (edit trips.collection_dict) for', df.activity_id)
                    exit()
            if len(trip_collections) == 1:
                for df in df_list:
                    for collection in tr.trip_dicts[df.activity_id]['collections']:
                        if collection in tr.collection_dict and not collection in ['europe','asia','Balkans'] and not 'Europe' in collection:
                            df.collection       = collection
                            df['collection']    = collection
                            break

            df_list, i_coord0_list = gpxr.sort_trips(df_list,  coord0=(90.0, 135.0))

            x0 = 0
            elevation_gain = 0
            for df, i_coord0 in zip(df_list, i_coord0_list):
                if i_coord0 == 0:
                    x = (x0 + df['cumulative_distance']) * 1e-3
                    y = df['altitude']
                else:
                    x = (x0 + df['cumulative_distance'].iloc[-1] -
                         df[::-1]['cumulative_distance']) * 1e-3
                    y = df[::-1]['altitude']

                df['x0'] = x0
                df['i_coord0'] = i_coord0
                x0 += df['cumulative_distance'].iloc[-1]
                elevation_gain += df['cumulative_elevation'].iloc[-1]
                df['trip_distance'] = x
                df['trip_altitude'] = y

            activity_dict_cols = [k for k in activity_dict[activity_ids[0]].keys()]
            activity_dict_cols_clean = [cleanhtml(k) for k in activity_dict_cols]

            df_summary = pd.DataFrame(
                columns=['name', 'date', 'cumulative_distance', 'cumulative_elevation', 'max_elevation'] + activity_dict_cols_clean
                )
            for _idf, df in enumerate(df_list):
                #print(df.loc[0]['time'])
                try:
                    dt = datetime.strptime(df.loc[0]['time'], '%Y-%m-%d %H:%M:%S+00:00')
                except:
                    #print(df.loc[0]['time'])
                    dt = datetime.strptime(df.loc[0]['time'], '%Y-%m-%d %H:%M:%S')
                df_dict = {
                    'name': activity_dict[df.activity_id]['activity_name'],
                    'strava_link_embed':df.strava_link_embed,
                    'veloviewer_link_embed':df.veloviewer_link_embed,
                    'link_embeds':df.link_embeds,
                    'collections':', '.join(tr.trip_dicts[df.activity_id]['collections']),
                    'cumulative_distance': df['cumulative_distance'].iloc[-1] * 1e-3,
                    'cumulative_elevation': df['cumulative_elevation'].iloc[-1],
                    'elevation_gain': activity_dict[df.activity_id]['elevation_gain'],
                    'max_elevation': max(df['altitude']),
                    'date': dt.strftime('%Y-%m-%d'),
                    'countries': [c for c in np.unique(df['country_name']) if not c == 'unknown'],
                }

                df_summary = df_summary.append(df_dict, ignore_index=True)
                df['customdata'] = df[['activity_name', 'trip_distance',
                                       'trip_altitude', 'country_name']].values.tolist()

            with open(f'trips/{trip_label}_summary.pkl', "wb") as f:
                pickle.dump(df_summary, f)

            pause_dicts = []
            flag_dicts = []
            tripdistance_arr = []
            postext_dicts = []
            pos = 0
            delta = 500

            big_df = pd.concat(df_list)
            big_df = big_df.sort_values(by=['trip_distance'])
            _i_old = 0; collection_old = ''


            travel_dicts = []
            _i = 0




            with open(f'trips/{trip_label}.pkl', "wb") as f:
                pickle.dump(big_df, f)

            last_country = ''
            # print(big_df.columns)
            last_time = datetime.strptime('2000-01-01 00:00:00+00:00', '%Y-%m-%d %H:%M:%S+00:00')
            for _i, row in big_df.iterrows():
                if row['trip_distance'] > pos:
                    postext_dicts.append({
                        'latitude': row['latitude'],
                        'longitude': row['longitude'],
                        'trip_distance': row["trip_distance"],
                        'altitude': row["altitude"],
                        'pos': pos,
                        'customdata': row['customdata'],
                    })
                    pos += delta
                if len(row['time']) == len('2022-11-18 19:58:47'):
                    row['time'] += '+00:00'
                curr_time = datetime.strptime(row['time'], '%Y-%m-%d %H:%M:%S+00:00')

                diff_time = min(abs((curr_time - last_time).seconds),
                                abs((last_time - curr_time).seconds))
                if diff_time > 15 * 60 and diff_time < 6 * 60 * 60:
                    # pause_arr_lat.append(row['latitude'])
                    # pause_arr_long.append(row['longitude'])
                    pause_dicts.append({
                        'latitude': row['latitude'],
                        'longitude': row['longitude'],
                    })
                last_time = curr_time
                if not row['country_iso'] == 'unknown':
                    if (not row['country_iso'] == last_country):
                        if row['country_iso'] == row['country_iso']:
                            flag_dicts.append({
                                'latitude': row['latitude'],
                                'longitude': row['longitude'],
                                'trip_distance': row["trip_distance"],
                                'altitude': row["altitude"],
                                'flag_dir': f"circle-flags/flags/{row['country_iso'].lower()}.svg.png",
                            })
                        last_country = row['country_iso']
            collection_data = {
                'df_list'       :df_list,
                'big_df'        :big_df,
                'df_summary'    :df_summary,
                'activity_dict' :activity_dict,
                'flag_dicts'    :flag_dicts,
                'pause_dicts'   :pause_dicts,
                'postext_dicts' :postext_dicts,
            }

            #print('dumping :OOO')
            with open(dfcollection_filename, "wb") as f:
                pickle.dump(collection_data, f)

        # with open(dfcollection_filename,'rb') as f2:
        #     collection_data = pickle.load(f2)
        #print(collection_data.keys())
        df_list       = collection_data['df_list'      ]
        big_df        = collection_data['big_df'       ]
        df_summary    = collection_data['df_summary'   ]
        activity_dict = collection_data['activity_dict']
        flag_dicts    = collection_data['flag_dicts'   ]
        pause_dicts   = collection_data['pause_dicts'  ]
        postext_dicts = collection_data['postext_dicts']

        #print(activity_dict)

        # ------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------
        # fig = make_subplots(rows=2, cols=1,
        #                     specs=[
        #                         [{"type": "scatter"}],
        #                         [{"type": "scattermapbox"}]
        #                     ]
        #                     )

        hovertemplate = "<br>".join([
            "%{customdata[0]}",
            "Distance: %{customdata[1]:.1f} km",
            "Altitude: %{customdata[2]:.1f} m",
            "Country: %{customdata[3]}<extra></extra>",
        ])
        hoverlabel = dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        )
        fig_altitude = go.Figure()
        for _idf, df in enumerate(df_list):
            #color = css_colors[_idf]
            color = cols.make_plot_cols(len(df_list))[_idf]
            #print(color, activity_dict[df.activity_id]['activity_name'])
            fig_altitude.add_trace(go.Scatter(
                x=df['trip_distance'], y=df['trip_altitude'],
                customdata=df['customdata'],
                name=activity_dict[df.activity_id]['activity_name'],
                showlegend=False,
                fill='tozeroy',
                marker=dict(
                    # size=14
                    color=color,
                ),
            ),
            )
        for flag_dict in flag_dicts:
            fig_altitude.add_layout_image(
                dict(
                    source=Image.open(flag_dict['flag_dir']),
                    xref="x",
                    yref="y",
                    xanchor="center",
                    yanchor="middle",
                    x=flag_dict["trip_distance"],
                    y=flag_dict["altitude"],
                    sizex=np.amax(big_df['trip_altitude']) * 0.2,
                    sizey=np.amax(big_df['trip_altitude']) * 0.2,
                    sizing="contain",
                    opacity=1.0,
                    layer="above"
                )
            )
        fig_altitude.update_traces(hovertemplate=hovertemplate)
        fig_altitude.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=20, b=20),
            hoverlabel=hoverlabel,
            #hovermode="x unified",
            xaxis_title="Distance (km)",
            yaxis_title="Elevation (m)",
            legend_title="Bike ride",
            # autosize=True,
        )
        fig_altitude.write_html("plotly_altitude.html")
        if bool_update_images:
            fig_altitude.write_image(f"images/{trip_label}_altitude.png")
        # ----------------------------------------------------------- #
        fig_map = go.Figure()
        for _idf, df in enumerate(df_list):
            color = cols.make_plot_cols(len(df_list))[_idf]
            fig_map.add_trace(
                go.Scattermapbox(
                    lat=df["latitude"], lon=df["longitude"],
                    mode='lines',
                    name=activity_dict[df.activity_id]['activity_name'],
                    customdata=df['customdata'],
                    showlegend=False,
                    marker=dict(
                        # size=14
                        color=color,
                    ),
                )
            )



        for flag_dict in flag_dicts:
            fig_altitude.add_layout_image(
                dict(
                    source=Image.open(flag_dict['flag_dir']),
                    xref="x",
                    yref="y",
                    xanchor="center",
                    yanchor="middle",
                    x=flag_dict["trip_distance"],
                    y=flag_dict["altitude"],
                    sizex=np.amax(big_df['trip_altitude']) * 0.2,
                    sizey=np.amax(big_df['trip_altitude']) * 0.2,
                    sizing="contain",
                    opacity=1.0,
                    layer="above"
                )
            )
        fig_map.add_trace(
            go.Scattermapbox(
                lat=[postext['latitude'] for postext in postext_dicts],
                lon=[postext['longitude'] for postext in postext_dicts],
                # mode='lines',
                customdata=[postext['customdata']
                            for postext in postext_dicts],
                mode="markers+text",
                text=[ f"{postext['pos']} km" for postext in postext_dicts],
                textposition="middle right",  # 'top center'
                #name=activity_dict[df.activity_id]['activity_name'],
                showlegend=False,
                marker=dict(
                    size=10,
                    color='red',
                )
            )
        )

        if bool_plot_pauses:
            pause_arr_lat = [pd['latitude'] for pd in pause_dicts]
            pause_arr_long = [pd['longitude'] for pd in pause_dicts]
            fig_map.add_trace(
                go.Scattermapbox(
                    lat=pause_arr_lat, lon=pause_arr_long,
                    # mode='lines',
                    mode="markers+text",
                    text='PAUSE!',
                    textposition="middle right",  # 'top center'
                    name='pauses',
                    showlegend=False,
                    visible=True,
                    marker=dict(
                        size=5,
                        color='green',
                    )
                )
            )
        #print(lat,lon,zoom)
        # print('!!!')
        # print(min(big_df["latitude"]),max(big_df["latitude"]), min(big_df["longitude"]),max(big_df["longitude"]))

        if bool_auto_zoom:
            zoom, center = zoom_center(
                    lons=[min(big_df["longitude"]),max(big_df["longitude"])],
                    lats=[min(big_df["latitude"]),max(big_df["latitude"])],
                    width_to_height = 4.0,
                )
            #print(zoom, center)
        else:
            center = {'lat':lat,'lon':lon}
        #print(big_df.columns)

        collections = set([df.collection for df in df_list])
        for collection in collections:
            df = big_df[big_df['collection'] == collection]
            for iletter,letter in enumerate(tr.collection_dict[collection]["name"]):

                z, c = zoom_center(
                        lons=[min(df["longitude"]),max(df["longitude"])],
                        lats=[min(df["latitude"]), max(df["latitude"])],
                        width_to_height = 4.0,
                    )

                dist_y = 180 / (2**zoom)


                letter_height = 0.1 * dist_y
                letter_width = 0.05 * dist_y

                x0 =  c['lon'] + iletter * letter_width
                y0 =  c['lat']

                if letter == ' ':
                    continue
                for contour in th.contour_dict[letter]:
                    fig_map.add_trace(
                        go.Scattermapbox(
                            lon = contour[0] * letter_width  + x0,
                            lat = contour[1] * letter_height + y0,
                            mode='lines',
                            name='infobox',
                            showlegend=False,
                            marker=dict(
                                # size=14
                                color='red',
                            ),
                        )
                    )

        fig_map.update_layout(
            #hovermode='closest',
            hovermode="x unified",
            autosize=True,
            mapbox=dict(
                # 'open-street-map',  # 'light',#'satellite',#'carto-positron',#'open-street-map',
                style='outdoors',
                accesstoken=apik.MAPBOX_TOKEN,
                domain={'x': [0.0, 1.0], 'y': [0.0, 1.0]},
                bearing=0,
                center = center,
                # center=dict(
                #     lat=lat,
                #     lon=lon
                # ),
                pitch=0,
                zoom=zoom,
            ),
            margin=dict(l=70, r=20, t=20, b=20),
            hoverlabel=hoverlabel,
            height=600,
        )
        # fig_map.update_geos(projection_type="sinusoidal")  # not working
        #fig_map.update_geos(projection_type="orthographic")
        fig_map.update_traces(hovertemplate=hovertemplate)
        #fig_map.update_layout(
        #    #height=600,width=2000,
        #    # "xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},
        #    # "yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},
        #    # "scene":{
        #    #     "xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},
        #    #     "yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},
        #    #     "zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}
        #    #     },
        #    # "shapedefaults":{"line":{"color":"#2a3f5f"}},
        #    # "annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},
        #    # "geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},
        #    # "title":{"x":0.05},
        #    # "mapbox":{"style":"light"}}},
        #    # "mapbox":{"center":{"lat":23,"lon":120},
        #    # "style":"outdoors",
        #    # "accesstoken":"pk.eyJ1IjoibWF4NjE0IiwiYSI6ImNsYXRjYnZvNTA2aTMzcHM1cTVwZ2lteWcifQ.uiBVHy3-95yEIzrjloixCA","bearing":0,"pitch":0,"zoom":3},
        #    # "hovermode":"x unified",
        #    # "autosize":false,
        #    # "margin":{"l":70,"r":20,"t":20,"b":20},
        #    # "hoverlabel":{"font":{"size":16,"family":"Rockwell"},"bgcolor":"white"}
        #    # },
        #    # {"responsive": true}
        #)
        fig_map.write_html("plotly_map.html")
        if bool_update_images:
            fig_map.write_image(f"images/{trip_label}_map.png")
            os.system(f'convert images/{trip_label}_map.png -crop 610x560+70+20 images/{trip_label}_map_cropped.png')
        # ['name', 'date', 'cumulative_distance', 'cumulative_elevation',
        #    'max_elevation', 'activity_id', 'activity_date', 'activity_name',
        #    'activity_type', 'activity_description', 'elapsed_time', 'distance',
        #    'max_heart_rate', 'relative_effort', 'commute', 'activity_gear',
        #
        #    'filename', 'athlete_weight', 'bike_weight', 'elapsed_time1',
        #    'moving_time', 'distance1', 'max_speed', 'average_speed',
        #    'elevation_gain', 'elevation_loss', 'elevation_low', 'elevation_high',
        #    'max_grade', 'average_grade', 'average_positive_grade',
        #    'average_negative_grade', 'max_cadence', 'average_cadence',
        #    'max_heart_rate1', 'average_heart_rate', 'max_watts', 'average_watts',
        #    'calories', 'max_temperature', 'average_temperature',
        #    'relative_effort1', 'total_work', 'number_of_runs', 'uphill_time',
        #    'downhill_time', 'other_time', 'perceived_exertion', 'type',
        #    'start_time', 'weighted_average_power', 'power_count',
        #    'prefer_perceived_exertion', 'perceived_relative_effort', 'commute1',
        #    'total_weight_lifted', 'from_upload', 'grade_adjusted_distance',
        #    'weather_observation_time', 'weather_condition', 'weather_temperature',
        #    'apparent_temperature', 'dewpoint', 'humidity', 'weather_pressure',
        #    'wind_speed', 'wind_gust', 'wind_bearing', 'precipitation_intensity',
        #    'sunrise_time', 'sunset_time', 'moon_phase', 'bike', 'gear',
        #    'precipitation_probability', 'precipitation_type', 'cloud_cover',
        #    'weather_visibility', 'uv_index', 'weather_ozone', 'jump_count',
        #    'total_grit', 'avg_flow', 'flagged', 'avg_elapsed_speed',
        #    'dirt_distance', 'newly_explored_distance',
        #    'newly_explored_dirt_distance', 'sport_type', 'filepath',
        #    'strava_link']
        # Create table
        #for l in df_summary['link_embeds']:
        #    print(l)
        header_vals = ['Name', 'Date', 'Distance (km)', 'Elevation gain (m)', 'Max elevation (m)','Links', 'Collections']
        header_cells_dict = {
                        'Name': df_summary.name,
                        'Date': df_summary.date,
                        'Distance (km)': df_summary.cumulative_distance,
                        'Elevation gain (m)': df_summary.elevation_gain,
                        'Max elevation (m)': df_summary.max_elevation,
                        'Links': df_summary.link_embeds,
                        #'Collections': df_summary.collections,
                    }
        header_vals = [k for k in header_cells_dict.keys()]
        cell_vals   = [header_cells_dict[k] for k in header_cells_dict.keys()]
        #
        fig = go.Figure(data=[go.Table(
            header=dict(values=header_vals,
                        fill_color='lightgray',
                        align='left'),
            cells=dict(values=cell_vals,
                       fill_color='lavender',
                       align='left',
                       format=["", "", "d", "d", "d", ""],
                       )
                    )
            ])
        fig.write_html("plotly_table.html")


        # COMBINE THE html files start
        #os.system('cat plotly_altitude.html plotly_map.html plotly_table.html > index.html')
        output_doc = BeautifulSoup()
        output_doc.append(output_doc.new_tag("html"))
        output_doc.html.append(output_doc.new_tag("body"))

        for file in ['plotly_altitude.html', 'plotly_map.html', 'plotly_table.html']:
            with open(file, 'r') as html_file:
                output_doc.body.extend(BeautifulSoup(html_file.read(), "html.parser").body)

        with open(index_filename,'w') as f:
            f.write(output_doc.prettify())
        os.system('rm plotly_altitude.html plotly_map.html plotly_table.html')

def main():
    selected_collections = [
        'europe',
        'asia',
        'hue-hcmc_2016',
        'singapore-kl_2017',
        'taiwan_2017',
        'yokohama-fukuoka_2019',
        'munich-dubrovnik_2021',
        'hamburg-paris_2022',
        'paris-geneva_2022',
        'hamburg-skagen_2022',
        'sion-munich_2022',
        'dubrovnik_istanbul_2022',
        # 'hike',
        # 'misc'
        ]
    for collection in selected_collections:
        plot_collection(collection,
                        bool_update_dfs         = True,
                        bool_update_triphtml    = False,     # Update trip html files
                        bool_update_images      = False,    # Update preview images for each trip
                        bool_update_csv         = False,    # Convert .gpx and .fit/.pkl --> csv
                        bool_update_pkl         = False,    # Convert .fit --> .pkl
                        bool_plot_pauses        = False,    # Plot pauses from the triphtml
                        bool_auto_zoom          = True,     # Auto zoom the map
                        )

    #
    with open('templates/INDEX_TEMPLATE3.html','r') as f:
        index_template = ''.join(f.readlines())
    for collection in selected_collections:
        with open(f'{collection}.html','w') as f:
            _out = index_template
            _out = _out.replace('<!--TOPNAV_LOCATION-->',get_topnav(selected_collections,active_collection = collection))
            _out = _out.replace('<!--TRIP_CONTENT-->',get_trip_content(collection))
            f.write(_out)
    with open('index.html','w') as f:
        _out = index_template
        _out = _out.replace('<!--TOPNAV_LOCATION-->',get_topnav(selected_collections,active_collection = 'index'))
        _str_content = '<ul id="rig">'
        for icollection, collection in enumerate(selected_collections):
            with open(f'trips/{collection}_summary.pkl','rb') as f2:
                summary_df = pickle.load(f2)

            cum_dist = round(sum(summary_df['cumulative_distance']))
            cum_alt  = round(sum(summary_df['elevation_gain']))
            cum_dur  = len(np.unique(summary_df['date'])) #(datetime.strptime(max(summary_df['date']),"%Y-%m-%d")  - datetime.strptime(min(summary_df['date']), "%Y-%m-%d")).days + 1
            countries = []
            for country_list in summary_df['countries']:
                for country in country_list:
                    countries.append(country)

            num_countries = len(np.unique(countries))
            if num_countries == 1:
                num_countries_str = '1 country'
            else:
                num_countries_str = f'{num_countries} countries'
            #print('num_countries',num_countries)
            _str_content +=f"""
                <li>
                    <a class="rig-cell" href="{collection}.html">
                        <img class="rig-img" src="images/{collection}_map_cropped.png">
                        <span class="rig-overlay"></span>
                        <span class="rig-text">{tr.collection_dict[collection]["name"]}<br> {cum_dur} days, {num_countries_str}<br>{cum_dist} km, {cum_alt} hm</span>
                    </a>
                </li>
            """
        _str_content += '</ul>'
        _out = _out.replace('<!--TRIP_CONTENT-->',_str_content)
        f.write(_out)
    #print(get_topnav(selected_collections,'hamburg-skagen_2022'))




main()
