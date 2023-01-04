# https://github.com/dtcooper/python-fitparse
import fitparse

import gpxpy
import gpxpy.gpx

import numpy as np
import math
import scipy.signal


import pickle
from os.path import exists

def import_data_gpx(filepaths):

    data_dicts = []
    for filepath in filepaths:
        print(f'reading {filepath}')
        gpx_file = open(filepath, 'r')
        gpx = gpxpy.parse(gpx_file)


        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    #print(f'Point at ({point.latitude},{point.longitude}) -> {point.elevation}')
                    pass

        data_dict = None
        data_dict = dict()

        data_dict['gpx'] = gpx


        data_dicts.append(data_dict)



    return data_dicts

def import_dict(filepath, bool_update_pkl = False):
    filepath_pkl = filepath.replace('.fit','.pkl')
    if bool_update_pkl or not exists(filepath_pkl):
        fit2pkl(filepath)
    with open(filepath_pkl, "rb") as f:
        data_dict =  pickle.load(f)
    return data_dict

def import_data(filepaths, bool_update_pkl = False):
    data_dicts = []
    for filepath in filepaths:
        print(f'reading {filepath}')
        data_dicts.append(import_dict(filepath,bool_update_pkl=bool_update_pkl))
    return data_dicts

def fit2pkl(filepath):
    fitfile = fitparse.FitFile(filepath)

    # Iterate over all messages of type "record"
    # (other types include "device_info", "file_creator", "event", etc)
    data_dict = None
    data_dict = dict()


    for irecord, record in enumerate(fitfile.get_messages("record")):
        #print(irecord,len(record))
        # Records can contain multiple pieces of data (ex: timestamp, latitude, longitude, etc)
        #print(record['speed'])
        for key in record:
            if 'timestamp' in key.name:
                timestamp = key.value
                if irecord == 0: start_time = timestamp
                elapsed_time = (timestamp - start_time).total_seconds()

        for i,data in enumerate(record):
            if not data.name in data_dict:
                data_dict[data.name] = [[],[]]
            data_dict[data.name][0].append(elapsed_time)
            data_dict[data.name][1].append(data.value)

            #if data.name == 'speed':
            #    speed_vec.append(data.value)
            #if data.name ==
            #pass
            # Print the name and value of the data (and the units if it has any)
            #if data.units:
            #    print(" * {}: {} ({})".format(data.name, data.value, data.units))
            #else:
            #    print(" * {}: {}".format(data.name, data.value))
            #print(data.value)
    data_dict['altitude_speed'] = [0]
    data_dict['elevation_gain'] = [0]
    data_dict['elapsed_time']   = [0]
    data_dict['dt']             = [0]
    data_dict['grade']          = [0]
    data_dict['dx']             = [0]

    N = 1e9
    #print(data_dict.keys())
    for key in ['timestamp','enhanced_speed','enhanced_altitude']:
        N = min(N, len(data_dict[key][0]))
    #    #print(key, )
    #print(data_dict['altitude'])
    #try:
    #N = len(data_dict['timestamp'][0])

    for i in range(1,N):
        #print(i, N, len(data_dict['timestamp'][1]))
        dt = max((data_dict['timestamp'][1][i] - data_dict['timestamp'][1][i-1]).total_seconds(), 1)
        dH = (data_dict['enhanced_altitude'][1][i]  - data_dict['enhanced_altitude'][1][i-1])
        dx = (data_dict['distance'][1][i]  - data_dict['distance'][1][i-1])



        data_dict['elapsed_time'].append((data_dict['timestamp'][1][i] - data_dict['timestamp'][1][0]).total_seconds())
        data_dict['dt'].append(dt)
        data_dict['dx'].append(dx)

        if dH > 0:
            data_dict['elevation_gain'].append(data_dict['elevation_gain'][i-1] + dH)
        else:
            data_dict['elevation_gain'].append(data_dict['elevation_gain'][i-1])
        #print(dH, dt)
        data_dict['altitude_speed'].append( dH / dt)

        if not data_dict['distance'][1][i] == data_dict['distance'][1][i-1]:
            data_dict['grade'].append( dH / dx )
        else:
            data_dict['grade'].append( 0 )

    data_dict['grade'] = np.array(data_dict['grade'])
    data_dict['savgol_grade'] = scipy.signal.savgol_filter(data_dict['grade'], 301, 5)

    #data_dict['moving_time'] = list(range(len(data_dict['elapsed_time'])))
    #data_dict['elapsed_minus_moving_time'] = [data_dict['elapsed_time'][i] - i for i in range(len(data_dict['elapsed_time']))]

    data_dict['moving_time'] = [0]
    min_break = 11 # seconds
    for i in range(1,len(data_dict['elapsed_time'])):
        #print(i)
        if data_dict['elapsed_time'][i]-data_dict['elapsed_time'][i-1] < min_break:
            data_dict['moving_time'].append(data_dict['moving_time'][i-1] + (data_dict['elapsed_time'][i]-data_dict['elapsed_time'][i-1]))
        else:
            data_dict['moving_time'].append(data_dict['moving_time'][i-1])

    #data_dict['moving_time'] = data_dict['elapsed_time']
    #data_dict['elapsed_minus_moving_time'] = [data_dict['moving_time'][i] - i for i in range(len(data_dict['moving_time']))]
    data_dict['elapsed_minus_moving_time'] = np.array(data_dict['elapsed_time']) - np.array(data_dict['moving_time'])
    data_dict['savgol_position_lat' ] = scipy.signal.savgol_filter(data_dict['position_lat'][1], 51, 3)
    data_dict['savgol_position_long'] = scipy.signal.savgol_filter(data_dict['position_long'][1], 51, 3)

    data_dict['angle'] = np.zeros(len(data_dict['savgol_position_long']))
    for i in range(len(data_dict['angle']) - 1):
        lat0  = data_dict['savgol_position_lat' ][i]
        lat1  = data_dict['savgol_position_lat' ][i + 1]
        long0 = data_dict['savgol_position_long'][i]
        long1 = data_dict['savgol_position_long'][i + 1]

        data_dict['angle'][i] = math.atan2(lat1-lat0, long1-long0)
    data_dict['angle'][-1] = data_dict['angle'][-2]
    data_dict['angle'] *= -360.0 / (2.0 * math.pi)
    data_dict['angle'] += 90
    #data_dict['angle'] = np.mod(data_dict['angle'],360)
    data_dict['filepath'] = filepath
    data_dict['file_str'] = filepath.split('/')[-1].replace('.fit','').replace('_',' ')
    #data_dicts.append(data_dict)

    filepath_pkl = filepath.replace('.fit','.pkl')
    dictionary_file = open(filepath_pkl, "wb")
    pickle.dump(data_dict, dictionary_file)
    dictionary_file.close()
    print('-->', filepath_pkl)


def calc_wind_adjusted_power(bike_speed=15, bike_dir=0, wind_speed=15, wind_dir=0):
    wind_angle = abs(bike_dir-wind_dir)
    alpha = math.radians(wind_angle)
    u = bike_speed; v = wind_speed
    w    = math.sqrt( (u + v * math.cos(alpha))**2 +(v * math.sin(alpha) )**2 )
    beta = math.acos( ( u + ( v * math.cos(alpha)) ) / w )
    d = w**2
    p = u * d * math.cos(beta)
    return p
#calc_wap()




def get_realistic_rider_speed(bike_speed=25, bike_dir=0, wind_speed=15, wind_dir=180):
    if bike_speed == 0: return 0
    power = calc_wind_adjusted_power(bike_speed=bike_speed, bike_dir=0, wind_speed=0, wind_dir=0)
    #print('power: ', power)
    for bike_speed in np.arange(0,100,0.1):
        p = calc_wind_adjusted_power(bike_speed=bike_speed, bike_dir=bike_dir, wind_speed=wind_speed, wind_dir=wind_dir)
        if p > power:
            #print(p,bike_speed)
            return bike_speed
    return 0

def kmph2mps(vel_kmh):
    return vel_kmh / 3.6
def mps2kmh(vel_mps):
    return vel_mps * 3.6


def get_gap(vel_in,grade_in):
    import scipy.interpolate
    data = {
        -30:115.32,
        -25:105.41,
        -20:94.51,
        -15:82.27,
        -10:68.05,
        -9:64.88,
        -8:61.58,
        -7:58.12,
        -6:54.49,
        -5:50.68,
        -4:46.64,
        -3:42.39,
        -2:37.90,
        -1:33.21,
        0:28.44,
        1:23.81,
        2:19.65,
        3:16.19,
        4:13.49,
        5:11.43,
        6:9.86,
        7:8.64,
        8:7.68,
        9:6.90,
        10:6.26,
        11:5.73,
        12:5.27,
        13:4.89,
        14:4.56,
        15:4.26,
        16:4.01,
        17:3.78,
        18:3.58,
        19:3.39,
        20:3.23,
        25:2.60,
        30:2.17,
        40:1.64,
        100:0.66
    }
    if grade_in < -0.30:
        grade_in = -0.30
    grades = np.array([d for d in data.keys()])
    speeds = np.array([d for d in data.values()]) / data[0]

    speeds_interp = scipy.interpolate.interp1d(grades, speeds)
    #print(grade_in, vel_in)
    try:
        return speeds_interp(grade_in * 100) * vel_in
    except:
        print(grade_in, vel_in)
    #print(vel_in, grades)
    #for i in range(len(grades)):
    #    #print(grades[i], grade_in)
    #    if grades[i] > grade_in * 100:
    #        return speeds[i] * vel_in
