from math import radians, cos, sin, asin, sqrt
import numpy as np

def poi_loader(filename):
    geodict = {}
    geodata = open(filename, 'r')
    for line in geodata.readlines():
        info = line.strip().split('\t')
        if len(info) == 4:
            geodict[info[0]] = (info[1],float(info[2]), float(info[3]))
    return geodict

def geo_distance(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6372.797 * c
    return km

def get_dist_matrix(filepath, poi2idx):
    geo_dict = poi_loader(filepath + '_PoiInfo.txt')
    dist_mat = np.zeros([len(geo_dict) + 1,len(geo_dict) + 1])
    dist_mat.fill(1000000.0)
    for i, p1 in enumerate(geo_dict):
        for j, p2 in enumerate(geo_dict):
            if i > j:
                continue
            dist_mat[ poi2idx[p1] ][ poi2idx[p2] ] = geo_distance( geo_dict[p1][1], geo_dict[p1][2],
                                                                   geo_dict[p2][1], geo_dict[p2][2])
            dist_mat[ poi2idx[p2] ][ poi2idx[p1] ] = dist_mat[ poi2idx[p1] ][ poi2idx[p2] ]

    return dist_mat
