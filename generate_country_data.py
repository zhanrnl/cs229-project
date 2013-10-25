import numpy as np
from math import *
from sklearn.neighbors import BallTree
import cPickle
import os
import hdf5_getters

base_dir = '../data/'
file_name = base_dir + 'allCountries.txt'
compact_file_name = base_dir + 'allCountriesCompact.txt'
sorted_coord_tuples_file = base_dir + 'sorted_coord_tuples.txt'

def country_lat_lng(line):
    ''' Given a line of the allCountries 1GB file, returns country code and 
    latitude and longitude in a tuple '''
    a = line.split('\t')
    return (a[8], float(a[4]), float(a[5]))

def build_compact_file():
    ''' Creates the file allCountriesCompact which contains only the information
    we need from the bigger allCountries file '''
    batchSize = 10000
    
    with open(file_name) as f:
        while True:
            thisBatch = []
            
            for i in xrange(batchSize):
                line = f.readline()
                if line == '':
                    break
                thisBatch.append(country_lat_lng(line))
                
            with open(compact_file_name, 'a') as f_out:
                for (country, lat, lng) in thisBatch:
                    f_out.write('{0}\t{1}\t{2}\n'.format(country, lat, lng))
                    
            if line == '':
                break

def build_coords_file():
    ''' WARNING: THIS IS VERY MEMORY INTENSIVE
    Converts the latitude/longitude data in the allCountriesCompact file into
    3d coordinate points paired with country codes. Then sorts these and writes
    to a file '''

    def extract(line):
        a = line.rstrip().split('\t')
        return (a[0], float(a[1]), float(a[2]))

    with open(compact_file_name) as f:
        
        coord_tuples = []
        for line in f:
            x = extract(line)
            coord_tuples.append((lat_long_to_3d(x[1], x[2]), x[0]))
    coord_tuples = sorted(coord_tuples)
    
    with open(sorted_coord_tuples_file, 'w') as f:
        for coord_tuple in coord_tuples:
            f.write(str(coord_tuple) + '\n')

def lat_long_to_3d(lat, lon):
    ''' Given latitude and longitude, returns equivalent Cartesian point '''
    lat = lat * pi / 180
    lon = lon * pi / 180

    x = cos(lat) * cos(lon)
    y = cos(lat) * sin(lon)
    z = sin(lat)

    return [x, y, z]
    
def build_BallTree():
    ''' Builds a scikit BallTree object from the latitudes and longitudes
    (and equivalent Cartesian points) in the compact file, so that we may 
    efficiently find the nearest point to a given query point '''
    list_of_points = []
    point_to_country = {}
    
    with open(compact_file_name) as f:
        for line in f:
            (country, lat, lng) = line.rstrip().split('\t')
            lat = float(lat)
            lng = float(lng)
            
            list_of_points.append(lat_long_to_3d(lat, lng))
            
    array_of_points = np.array(list_of_points)
    
    import gc; gc.collect()
            
    return array_of_points, BallTree(array_of_points)
    
def lookup_loc_BallTree(lat, lng, ball_tree, array_of_points):
    ''' Wrapper for querying the BallTree (given latitude and longitude, returns
    the closest Cartesian value '''    
    distance, index = ball_tree.query(lat_long_to_3d(lat, lng))
    return array_of_points[index[0][0]].tolist()   
        
def get_countries_from_coords(coord_key_tuples_list):
    ''' Given a list of tuples that look like ([x, y, z], KEY), does a batch
    lookup of the country codes corresponding to these values of [x, y, z] '''
    coord_key_tuples_list = sorted(coord_key_tuples_list)
    
    output = []
    
    with open(sorted_coord_tuples_file, 'r') as f:
        i = 0
        
        curr_file_elem = eval(f.readline())
        
        while i < len(coord_key_tuples_list) and f.readline() != '':
            
            while curr_file_elem[0] != coord_key_tuples_list[i][0]:
                curr_file_elem = eval(f.readline())
            
            while curr_file_elem[0] == coord_key_tuples_list[i][0]:
                output.append((coord_key_tuples_list[i][1], curr_file_elem[1]))
                i += 1
                
                if i == len(coord_key_tuples_list):
                    break

    return output
    
def bulk_lookup_loc(lat_lng_key_tuples_list, ball_tree, array_of_points):
    ''' Given a list of (lat, lng, KEY) tuples, returns a list of tuples that
    look like (KEY, Country Code) '''
    def elem_to_result(x):
        return (lookup_loc_BallTree(x[0], x[1], ball_tree, array_of_points), x[2])
    
    coord_key_tuples_list = [elem_to_result(x) for x in lat_lng_key_tuples_list]
    
    return get_countries_from_coords(coord_key_tuples_list)

def get_all_filepaths(indir):
    ''' Given a directory, return all .h5 files in this directory and its
    subdirectories '''
    files = []

    for root, dirs, filenames in os.walk(indir):
        for f in filenames:
            if os.path.splitext(f)[1] == '.h5':
                files.append(os.path.join(root, f))

    return files

def save_filenames_and_artists():
    ''' Iterates through all files in the dataset and saves lists of tuples
    that look like (lat, lng, filename) and (lat, lng, artist_name) '''
    files = get_all_filepaths(base_dir + 'MillionSongSubset/')
    
    filename_list = []
    artist_list = [] 

    for f in files:
        h5 = hdf5_getters.open_h5_file_read(f)
        
        artist_name = hdf5_getters.get_artist_name(h5)
        lat = hdf5_getters.get_artist_latitude(h5)
        lng = hdf5_getters.get_artist_longitude(h5)

        h5.close()

        if not isnan(lat):
            filename_list.append((lat, lng, f))
            artist_list.append((lat, lng, artist_name))

    cPickle.dump(filename_list, open(base_dir + 'filename_list', 'wb'))
    cPickle.dump(artist_list, open(base_dir + 'artist_list', 'wb'))

def save_filename_to_country_dict(ball_tree, array_of_points):
    ''' Saves a dictionary of Filename => Country code '''
    filename_list = cPickle.load(open(base_dir + 'filename_list', 'rb'))
    
    filename_country_list = bulk_lookup_loc(filename_list, ball_tree, array_of_points)

    filename_country_dict = {}
    for (k, v) in filename_country_list:
        filename_country_dict[k] = v
    
    cPickle.dump(filename_country_dict, open(base_dir + 'filename_country_dict', 'wb'))
    return filename_country_dict

def benchmark(n, ball_tree, array_of_points):
    import random
    import time
    
    print 'Starting benchmark. Going to create random pairs now'
    
    def rand_pair(i):
        return (180 * (random.random() - 0.5), 90 * (random.random() - 0.5), '#' + str(i))
        
    lat_lng_key_tuples_list = [rand_pair(i) for i in xrange(n)]
    
    print 'About to start the bulk lookup...'
    start = time.time()
    results = bulk_lookup_loc(lat_lng_key_tuples_list, ball_tree, array_of_points)
    elapsed = time.time() - start 
    
    num_not_none = sum(x != None for x in results)
    pnn = 100 * float(num_not_none) / n
    
    print 'Total time: {0} seconds'.format(elapsed)
    print 'Requests per second: {0}'.format(n / elapsed)
    print 'Percentage where we got an actual country (not None): {0}%'.format(pnn)
    
if __name__ == '__main__':
    print 'Starting to save filenames and artists'
    save_filenames_and_artists()
    print 'Done saving filenames and artists'
    
    # build_compact_file()
    # build_coords_file()
    print 'Building BallTree'
    array_of_points, ball_tree = build_BallTree()
    print 'Done building BallTree'
    # benchmark(10, ball_tree, array_of_points)

    print 'Saving filename => country code dictionary'
    save_filename_to_country_dict(ball_tree, array_of_points)
    print 'Done with all tasks. Exiting...'
