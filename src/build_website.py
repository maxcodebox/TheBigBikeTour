import re
import trips as tr
import numpy as np
import pickle




# as per recommendation from @freylis, compile once only
CLEANR = re.compile('<.*?>')

def cleanhtml(raw_html):
    cleantext = re.sub(CLEANR, '', raw_html)
    return cleantext


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
    with open(f'figures/html/{collection}.html','r') as f:
        return ''.join(f.readlines())

def emoji_to_html(emoji):
    # Convert each character in the flag emoji to its Unicode code point
    code_points = [ord(char) for char in emoji]
    
    # Convert code points to HTML entities
    html_entities = ''.join(f'&#x{code_point:X};' for code_point in code_points)
    
    return html_entities

def main():


    selected_collections = ['norway-turkey','berlin-tarifa','taiwan_2017','hue-hcmc_2016','yokohama-fukuoka_2019','bavarian-alp-traverse']
    #selected_collections = ['norway-turkey']
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
            
            with open(f'data/{collection}_summary.pkl','rb') as f2:
                collection_summary = pickle.load(f2)
            #print(collection_summary)
            days = collection_summary['days']
            
            flags_html = ''.join([emoji_to_html(flag) for flag in collection_summary['flags']])
            _str_content +=f"""
                <li>
                    <a class="rig-cell" href="{collection}.html">
                        <img class="rig-img" src="figures/static/{collection}_map.png">
                        <span class="rig-overlay"></span>
                        <span class="rig-text" style="position: absolute; top: 30px; left: 10px;">
                            {emoji_to_html(tr.collection_dict[collection]["name"])}<br>
                            {days} days<br>
                            {collection_summary['distance_km']:.0f} km, {collection_summary['elevation_gain_m']:.0f} hm, {collection_summary['moving_time_h']:.0f} hours<br>
                            {collection_summary['distance_km']/days:.0f} km/day, {collection_summary['elevation_gain_m']/days:.0f} hm/day, {collection_summary['moving_time_h'] / days:.1f} hours/day<br>
                            {len(collection_summary['flags'])} countries ({flags_html})
                        </span>
                    </a>
                </li>
            """
        _str_content += '</ul>'
        _out = _out.replace('<!--TRIP_CONTENT-->',_str_content)
        f.write(_out)
    print('open index.html')


if __name__ == '__main__':
    main()