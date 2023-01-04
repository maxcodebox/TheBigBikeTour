import pandas as pd
from pathlib import Path
import geopy.distance
import gpxpy
import countries  # https://github.com/che0/countries
import fitplotlib as fpl
# pip install --global-option=build_ext --global-option="-I/usr/include/gdal" GDAL==`gdal-config --version`
# http://thematicmapping.org/downloads/world_borders.php
#cc = countries.CountryChecker('world_borders/TM_WORLD_BORDERS_SIMPL-0.3.shp')
#cc = countries.CountryChecker('world_borders/TM_WORLD_BORDERS-0.3.shp')

def read_gpx(filepath, bool_update_csv=False,bool_update_pkl=False,activity_id=None):
    print(f'Reading {filepath}')
    # fp = filepath.replace('/gpx/', '/csv/')
    # fp = fp.replace(fp.split('/')[-1], '')
    # Path(fp).mkdir(parents=True, exist_ok=True)

    #filepath_out = filepath.replace('/gpx/', '/csv/').replace('.gpx', '.csv')
    filepath_out = filepath.replace('.gpx', '.csv').replace('.fit', '.csv')
    if not Path(filepath_out).is_file():
        bool_update_csv = True
    if Path(filepath_out).is_file() and not bool_update_csv:
        df = pd.read_csv(filepath_out)
        if not 'country_iso' in df.columns:
            bool_update_csv = True




    if bool_update_csv:
        conv2deg = 1.0 / (2**32 / 360)
        if '.fit' in filepath:
            #print('YOOO')
            d = fpl.import_dict(filepath, bool_update_pkl = bool_update_pkl)
            data = []

            for long,lat,alt,time,speed in zip( d['position_long'][1], d['position_lat'][1], d['enhanced_altitude'][1], d['timestamp'][1], d['enhanced_speed'][1]):
                if not None in [long,lat,alt,time,speed]:
                    data.append([long * conv2deg, lat * conv2deg, alt, time, speed])
        else:
            gpx = gpxpy.parse(open(filepath))
            track = gpx.tracks[0]
            segment = track.segments[0]
            #
            data = []
            segment_length = segment.length_3d()
            for point_idx, point in enumerate(segment.points):
                data.append([point.longitude, point.latitude,
                             point.elevation, point.time, segment.get_speed(point_idx)])
        columns = ['longitude', 'latitude', 'altitude', 'time', 'speed']
        df = pd.DataFrame(data, columns=columns)
        df = df.dropna()
        cord = [(row['latitude'], row['longitude'])
                for _index, row in df.iterrows()]
        df['distance'] = [
            0] + [geopy.distance.distance(from_, to).m for from_, to in zip(cord[:-1], cord[1:])]
        # Cumulative distance.
        df['cumulative_distance'] = df['distance'].cumsum()
        #
        elevation_gain = 0
        cumulative_elevation = [elevation_gain]
        for i in range(1, len(df)):
            elevation_gain += max(0, df['altitude'].iloc[i] -
                                  df['altitude'].iloc[i - 1])
            cumulative_elevation.append(elevation_gain)
        df['cumulative_elevation'] = cumulative_elevation

        df['country_name'] = 'unknown'
        df['country_iso'] = 'unknown'
        delta_i = 1000
        for _i, row in df.iterrows():
            if _i % delta_i == 0 and _i > 0:
                country = cc.getCountry(countries.Point(
                    row['latitude'], row['longitude']))
                if not country is None:
                    df.at[_i - delta_i:_i, 'country_name'] = str(country)
                    df.at[_i - delta_i:_i, 'country_iso'] = country.iso

        #df.at[-(len(df) % delta_i):,'country'] = country.iso
        df.to_csv(filepath_out)
    # print(filepath_out,bool_update_csv)
    df = pd.read_csv(filepath_out)
    df.columns = [c.lower().replace(' ','_') for c in df.columns]
    df = df.iloc[::15]

    # alt = list(df['altitude']); dist = list(df['distance'])
    # for i in range(1,len(dist)):
    #     print('--',dist[i],alt[i])
    #     print((alt[i]-alt[i-1]) /  (dist[i] - dist[i-1]))
    # df['grade'] = [0] + [ (alt[i]-alt[i-1]) /  (dist[i] - dist[i-1]) for i in range(1,len(dist))]
    #print(activity_id)
    df.activity_id = activity_id
    df['activity_id'] = activity_id
    df.filename = filepath.split('/')[-1].replace('_', ' ').replace('.gpx', '')
    df['filename'] = filepath.split('/')[-1].replace('_', ' ').replace('.gpx', '')
    df['x0'] = 0
    # print(set(df['country']))
    return df


def get_closest_ride(df_list, coord0=(90.0, 135.0)):
    i_closest = 0
    i_coord0 = 0
    min_distance = 10e10
    for idf, df in enumerate(df_list):
        for ip in [0, -1]:
            starting_point = (df['latitude'].iloc[ip],
                              df['longitude'].iloc[ip])
            try:
                distance_from_coord0 = geopy.distance.distance(starting_point, coord0).m
                if distance_from_coord0 < min_distance:
                    min_distance = distance_from_coord0
                    i_closest = idf
                    i_coord0 = ip
            except:
                pass
    return i_closest, i_coord0


def sort_trips(df_list, coord0=(90.0, 135.0)):
    df_list_out = []
    i_coord0_list_out = []
    i = 0
    while len(df_list) > 0:
        i_closest, i_coord0 = get_closest_ride(df_list, coord0=coord0)
        df_list_out.append(df_list[i_closest])
        #
        i_coord0_list_out.append(i_coord0)
        #
        i_coord0 = -1 * (i_coord0 + 1)  # 0 -> -1, -1 -> 0
        coord0 = (df_list[i_closest]['latitude'].iloc[i_coord0],
                  df_list[i_closest]['longitude'].iloc[i_coord0])
        #
        df_list.pop(i_closest)
        #
        i += 1
        #
    return df_list_out, i_coord0_list_out
