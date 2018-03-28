import datetime
import random
import numpy as np
from collections import defaultdict
from collections import Counter
import networkx as nx
import math
import time
from math import log1p
from math import radians, cos, sin, asin, sqrt
import powerlaw
from sklearn import preprocessing
#from pygraph.classes.digraph import digraph
#import Geohash as Gy
#import geohash as gh
import csv
from ast import literal_eval

UNCLASSIFIED = False
NOISE = 0


def poi_subgraph(poi_graph, grid_poi_list):
    gpg = poi_graph.subgraph(grid_poi_list)
    return gpg

    
'''
def gen_geohash(filepath, dataset, precision):
    geo_dict = poi_loader(filepath + dataset + '_PoiInfo.txt')
    gh_file = open(filepath + dataset + '_GeoHash_' + str(precision) + '.txt', 'w')
    for p in geo_dict:
        cate, lng, lat = geo_dict[p]
        code = Gy.encode(lng, lat, precision=precision)
        gh_file.write(str(p) + '\t' + str(code) + '\n')
    gh_file.close()
'''
def geohash_loader(filepath, dataset, precision):
    gh_dict = {}
    gh_file = open(filepath + dataset + '_GeoHash_' + str(precision) + '.txt', 'r')
    for line in gh_file.readlines():
        info = line.strip().split('\t')
        gh_dict[info[0]] = info[1]
    gh_file.close()
    
    return gh_dict

    
def dist_filter(current_poi, geo_dict, dist):
    candidate_poi = []
    [cate1,lng1,lat1] = geo_dict[current_poi]
    for p in geo_dict.keys():
        [cate2,lng2,lat2] = geo_dict[p]
        if geo_distance(lng1, lat1, lng2, lat2) <= dist:
            candidate_poi.append(p)
    return candidate_poi


def visited_dict(checkin_list):
    user_checkin_dict = listtodict(checkin_list, 1)
    visited = defaultdict(list)
    for u in user_checkin_dict.keys():
        visited[u] = user_checkin_dict[u].keys()
    return visited



def gen_test_query(testing_time_successive):
    
    temp_list = []
    for user in testing_time_successive:
        for q in testing_time_successive[user]:
            #u, ans, ans_poi, l_tm1, cur_poi
            q1, q2, q3 = q  # context, ans_poi, ans
            if len(q1)!=0 and len(q3)!=0:
                u = user
                ans = set(q3)
                ans_poi = q2
                cur_poi = q1[-1]
                l_tm1 = set(q1)
                temp_list.append((u,ans,ans_poi,l_tm1,cur_poi))
    
    return temp_list


def sigmoid(x):
    if x >= 0:
        return math.exp(-np.logaddexp(0, -x))
    else:
        return math.exp(x - np.logaddexp(x, 0))






def gen_poi_testing(datalist, te_list, ts):
    
    
    user_checked_dict = {}
    for q in datalist:
        (user, poi, day, time, categoty, lng ,lat) = q
        if user not in user_checked_dict:
            user_checked_dict[user] = []
        
        user_checked_dict[user].append((poi,day,time))

    
    poi_dict = {}    
    for q in te_list:
        (user, poi, day, time, categoty, lng ,lat) = q
        
        if poi not in poi_dict:
            poi_dict[poi] = []
        
        poi_dict[poi].append((user,day,time))    
    

    
    test_dict = {}
    for p in poi_dict:
  
        
        """
        if len(poi_dict[p]) == 1:
            if p not in test_dict:
                test_dict[p] = {}
            (u1, d1, t1) = poi_dict[p][0]
            test_dict[p][(d1,t1)] = [u1]
        
            
        i = 0
        if len(poi_dict[p])>1:
            while i<len(poi_dict[p]):
                (u1, d1, t1) = poi_dict[p][i]
                userset = [] 
                j = i+1
                while j<len(poi_dict[p]):
                    (u2, d2, t2) = poi_dict[p][j]
                    if abs(timediff(d1, t1, d2, t2) / float(3600)) <= ts:
                        userset.append(u2)
                        j = j+1
                        i = j
                    else:
                        i = j
                        break
                        
                if len(userset) > 0:
                    if p not in test_dict:
                        test_dict[p] = {}   
                    temp = []
                    temp.append(u1)
                    for u in userset:
                        temp.append(u)
                    test_dict[p][(d1,t1)] = temp
                else:
                    
                    if p not in test_dict:
                        test_dict[p] = {}  
                    test_dict[p][(d1,t1)] = [u1]
    
                if i == len(poi_dict[p])-1:
                    break
        """
        """
        if p not in test_dict:
            test_dict[p] = {}  
        """   
        i = 0
        while i<len(poi_dict[p]):
            (u1,d1,t1) = poi_dict[p][i]  
            userset = []
            j = i+1 
            while j<len(poi_dict[p]):
                (u2, d2, t2) = poi_dict[p][j]
                if abs(timediff(d1, t1, d2, t2) / float(3600)) <= ts:
                    userset.append(u2)
                    j = j+1
                    i = j
                else:
                    i = j
                    break
                
            if len(userset) > 0:
                if p not in test_dict:
                    test_dict[p] = {}   
                temp = []
                temp.append(u1)
                for u in userset:
                    temp.append(u)
                test_dict[p][(d1,t1)] = temp
            """
            else:
                    
                if p not in test_dict:
                    test_dict[p] = {}  
                test_dict[p][(d1,t1)] = [u1]
            """
    
            if i == len(poi_dict[p])-1:
                break
        """            
        for q in range(len(poi_dict[p])):
            (user, cur_day, cur_time) = poi_dict[p][q]

            temp_list = []
            temp_list.append(user)
            
            j = q+1    
            for j in range(j, len(poi_dict[p])):
                (j_u, j_day, j_time) = poi_dict[p][j]
                if abs(timediff(cur_day, cur_time, j_day, j_time)) / float(3600) <= ts:
                    temp_list.append(j_u)
                else:
                    break
                
            if len(temp_list)!=0:
                test_dict[p][(cur_day,cur_time)] =  temp_list       
        """
        
    return test_dict, user_checked_dict

def timediff2(date1, date2):
    return (date2 - date1).total_seconds()

def dataset_splitting2(filepath, dataset, train_percent, tune_percent):
    checkinlist = checkinstolist2(filepath + dataset + '_Checkins.txt')
    index_1 = int(round(len(checkinlist) * train_percent))
    index_2 = int(round(len(checkinlist) * (train_percent + tune_percent)))
    train = checkinlist[:index_1]
    tune = checkinlist[index_1:index_2]
    tests = checkinlist[index_2:]
    return train, tune, tests

def dataset_pair2(filepath, dataset, train_percent, tune_percent, query_limit, time_threshold, seq_length, load_seq):
    train_list, tune_list, test_list = dataset_splitting2(filepath, dataset, train_percent, tune_percent)

    
    training, tuning, testing = listtodict2(train_list, 1), listtodict2(tune_list, 0), listtodict2(test_list, 0)

    tune_dict = listtodict2(tune_list, 1)
    ans, tune_ans = defaultdict(list), defaultdict(list)
    for u in training.keys():
        visited = set(training[u].keys())
        if len(visited) == 0:
            continue
        #   Get Tuning Pair
        if u in tuning.keys():
            tuning_pair = []
            for i in range(len(tuning[u]) - 1):
                (p1, d1) = tuning[u][i]
                future = [] 
                context = [] 
                for j in range(i+1, len(tuning[u])):
                    (pa, da) = tuning[u][j]
                    if pa not in visited:
                        if timediff2(d1, da) / float(3600) <= time_threshold:
                            future.append((pa))
                        else:
                            break
                
            
                if len(future) != 0:   
                    for j in range(i-1, -1, -1):
                        pb, db = tuning[u][j]
                        if timediff2(db, d1) / float(3600) <= time_threshold:
                            context.append((pb))
                        else:
                            break
                        
                if len(context) >=load_seq  and len(future)!=0:
                    

                    current_poi = p1
                    current_time = d1
                    
                    if load_seq !=0:
                        seq = context[-load_seq:]
                        real_in = []
                        for k in seq:
                            (pi) = k
                            real_in.append((pi))
                        seq_set = set(real_in)
                    else:
                        seq_set = set()
                    
                    temp = []
                    for k in future:
                        (pj) = k
                        temp.append(pj)

                    ans_set = set(temp)
                    poi_ans = temp[0]
                    
                    
                    if len(tuning_pair) == 0:
                        tuning_pair.append((current_poi, current_time, seq_set, poi_ans, ans_set))
                    else:
                        if list(ans_set) > list(tuning_pair[-1][-1]):
                            tuning_pair.append((current_poi, current_time, seq_set, poi_ans, ans_set))
                            
                            
                    #tuning_pair.append((current_poi, current_time, seq_set, poi_ans, ans_set))
            
            if len(tuning_pair) >= query_limit:
                tune_ans[u] = random.choice(tuning_pair)
                
            visited |= set(tune_dict[u].keys())

        #   Get Testing Pair

        if u in testing.keys():
            testing_pair = []
            for i in range(len(testing[u]) - 1):
                (p1, d1) = testing[u][i]
                future = [] 
                context = [] 
                for j in range(i+1, len(testing[u])):
                    (pa, da) = testing[u][j]
                    if pa not in visited:
                        if timediff2(d1, da) / float(3600) <= time_threshold:
                            future.append((pa))
                        else:
                            break
                
            
                if len(future) != 0:   
                    for j in range(i-1, -1, -1):
                        pb, db = testing[u][j]
                        if timediff2(db, d1) / float(3600) <= time_threshold:
                            context.append((pb))
                        else:
                            break
                if len(context) >=load_seq  and len(future)!=0:
                    current_poi = p1
                    current_time = d1
                    
                    if load_seq !=0:
                        seq = context[-load_seq:]
                        real_in = []
                        for k in seq:
                            (pi) = k
                            real_in.append((pi))
                        seq_set = set(real_in)
                    else:
                        seq_set = set()
                
                    temp = []
                    for k in future:
                        (pj) = k
                        temp.append(pj)

                    ans_set = set(temp)
                    poi_ans = temp[0]
                    
                    #if len(list(set(real_in))) >1 :
                        #testing_pair.append((current_poi, current_time, seq_set, poi_ans, ans_set))
  
                    if len(testing_pair) == 0:
                        testing_pair.append((current_poi, current_time, seq_set, poi_ans, ans_set))
                    else:
                        if list(ans_set) > list(testing_pair[-1][-1]):
                            testing_pair.append((current_poi, current_time, seq_set, poi_ans, ans_set))
                    #testing_pair.append((current_poi, current_time, seq_set, poi_ans, ans_set))
            
            if len(testing_pair) >= query_limit:
                #ans[u] = random.sample(testing_pair, query_limit)
                ans[u] = testing_pair
               
            
              
            #if len(testing_pair)>= 1:     
                #ans[u] = testing_pair 
            #ans[u] = testing_pair
   
    print ('tune_case : ' + str(len(tune_ans.keys())))
    print ('test_case : ' + str(len(ans.keys())))
  
    return train_list, tune_list, ans, tune_ans


"""
def dataset_pair2(filepath, dataset, train_percent, tune_percent, query_limit, time_threshold):
    train_list, tune_list, test_list = dataset_splitting2(filepath, dataset, train_percent, tune_percent)
    training, tuning, testing = listtodict2(train_list, 1), listtodict2(tune_list, 0), listtodict2(test_list, 0)
    tune_dict = listtodict2(tune_list, 1)
    ans, tune_ans = defaultdict(list), defaultdict(list)
    for u in training.keys():
        visited = set(training[u].keys())
        if len(visited) == 0:
            continue
        #   Get Tuning Pair
        if u in tuning.keys():
            tuning_pair = []
            ans_set = set()
            seq_set = set()
            for i in range(len(tuning[u]) - 1):
                (p1, d1) = tuning[u][i]
                (p2, d2) = tuning[u][i + 1]
                if not ans_set:
                    start_time = d1
                    if p2 not in visited:
                        if timediff2(start_time, d2) / float(3600) <= time_threshold:
                            current_poi = p1
                            current_time = d1
                            seq_set = {p1}
                            ans_set |= {p2}
                            poi_ans = p2
                            for j in range(i - 1, -1, -1):
                                pj, dj = tuning[u][j]
                                if timediff2(dj, start_time) / float(3600) <= time_threshold:
                                    seq_set |= {pj}
                                else:
                                    break
                else:
                    if timediff2(start_time, d2) / float(3600) <= time_threshold:
                        if p2 not in visited:
                            ans_set |= {p2}
                            visited |= {p2}
                    else:
                        tuning_pair.append((current_poi, current_time, seq_set, poi_ans, ans_set))
                        ans_set = set()
                        seq_set = set()
            if len(tuning_pair) >= 1:
                tune_ans[u] = random.choice(tuning_pair)
            visited |= set(tune_dict[u].keys())

        #   Get Testing Pair
        if u in testing.keys():
            testing_pair = []
            ans_set = set()
            seq_set = set()
            for i in range(len(testing[u]) - 1):
                (p1, d1) = testing[u][i]
                (p2, d2) = testing[u][i + 1]
                if not ans_set:
                    start_time = d1
                    if p2 not in visited:
                        if timediff2(start_time, d2) / float(3600) <= time_threshold:
                            current_poi = p1
                            current_time = d1
                            seq_set = {p1}
                            ans_set |= {p2}
                            poi_ans = p2
                            for j in range(i - 1, -1, -1):
                                pj, dj = testing[u][j]
                                if timediff2(dj, start_time) / float(3600) <= time_threshold:
                                    seq_set |= {pj}
                                else:
                                    break
                else:
                    if timediff2(start_time, d2) / float(3600) <= time_threshold:
                        if p2 not in visited:
                            ans_set |= {p2}
                            visited |= {p2}
                    else:
                        testing_pair.append((current_poi, current_time, seq_set, poi_ans, ans_set))
                        ans_set = set()
                        seq_set = set()
            if len(testing_pair) >= query_limit:
                ans[u] = random.sample(testing_pair, query_limit)
    print 'tune_case : ' + str(len(tune_ans.keys()))
    print 'test_case : ' + str(len(ans.keys()))
    return train_list, tune_list, ans, tune_ans
"""
def testing_data_loading(filepath, dataset, time_threshold):
    tfile = open(filepath + dataset + '_Test_' + str(time_threshold) + '.txt', 'r')
    query_dict = defaultdict(list)
    # u, cur_poi, seq_set, ans_poi, ans_set
    for line in tfile.readlines():
        info = line.strip().split('\t')
        for i in range(len(info)):
            info[i] = info[i].strip().split(' ')
        for i in range(len(info[2])):
            info[2][i] = int(info[2][i])
        for i in range(len(info[4])):
            info[4][i] = int(info[4][i])
        query_dict[int(info[0][0])].append((int(info[1][0]), set(info[2]), int(info[3][0]), set(info[4])))
    tfile.close()
    return query_dict


def testing_data_loading_o(filepath, dataset, time_threshold):
    

    tfile = open(filepath + dataset + '_Test2_' + str(time_threshold) + '.txt', 'r')
    query_dict = defaultdict(list)
    # u, cur_poi, cur_time, seq_set, ans_poi, ans_set
    for line in tfile.readlines():
        info = line.strip().split('\t')   
        if len(info) == 5:
            for i in range(len(info)):
                info[i] = info[i].strip().split(' ')
            for i in range(len(info[4])):
                info[4][i] = str(info[4][i])
                
            query_dict[str(info[0][0])].append((str(info[1][0]), (info[2][0],info[2][1]), set('0'), str(info[3][0]), set(info[4])))
        else:
            for i in range(len(info)):
                info[i] = info[i].strip().split(' ')
   
            for i in range(len(info[3])):
                info[3][i] = str(info[3][i])
            for i in range(len(info[5])):
                info[5][i] = str(info[5][i])
        
            query_dict[str(info[0][0])].append((str(info[1][0]), (info[2][0],info[2][1]), set(info[3]), str(info[4][0]), set(info[5])))
    tfile.close()
    return query_dict



def tuneing_data_loading_o(filepath, dataset, time_threshold):
    

    tfile = open(filepath + dataset + '_Tune2_' + str(time_threshold) + '.txt', 'r')
    query_dict = defaultdict(list)
    # u, cur_poi, cur_time, seq_set, ans_poi, ans_set
    for line in tfile.readlines():
        info = line.strip().split('\t')   
        
        if len(info) == 5:
            for i in range(len(info)):
                info[i] = info[i].strip().split(' ')
            for i in range(len(info[4])):
                info[4][i] = str(info[4][i])
                
            query_dict[str(info[0][0])].append((str(info[1][0]), (info[2][0],info[2][1]), set('0'), str(info[3][0]), set(info[4])))
        else:
            for i in range(len(info)):
                info[i] = info[i].strip().split(' ')
   
            for i in range(len(info[3])):
                info[3][i] = str(info[3][i])
            for i in range(len(info[5])):
                info[5][i] = str(info[5][i])
        
            query_dict[str(info[0][0])].append((str(info[1][0]), (info[2][0],info[2][1]), set(info[3]), str(info[4][0]), set(info[5])))
    tfile.close()
    return query_dict


def tuning_data_gen(filepath, dataset, time_threshold, seq_length, load_seq):
    train_list, tune_list, ans, tune_ans = dataset_pair2(filepath, dataset, 0.7, 0.1, 3, time_threshold, seq_length, load_seq)
    tune_file = open(filepath + dataset + '_Tune2_' + str(time_threshold) + '.txt', 'w')
    for u in tune_ans.keys():
        current_poi, current_time, seq_set, ans_poi, ans_set = tune_ans[u]
        tune_file.write(str(u) + '\t')
        tune_file.write(str(current_poi) + '\t')
        tune_file.write(str(current_time) + '\t')
        for p in seq_set:
            tune_file.write(str(p) + ' ')
        tune_file.write('\t')
        tune_file.write(str(ans_poi))
        tune_file.write('\t')
        for p in ans_set:
            tune_file.write(str(p) + ' ')
        tune_file.write('\n')
    tune_file.close()

def testing_data_genfile(filepath, dataset, time_threshold, seq_length, load_seq):
    train_list, tune_list, ans, tune_ans = dataset_pair2(filepath, dataset, 0.7, 0.1, 3, time_threshold, seq_length, load_seq)
    test_file = open(filepath + dataset + '_Test2_' + str(time_threshold) + '.txt', 'w')
    for u in ans.keys():
        for current_poi, current_time, seq_set, ans_poi, ans_set in ans[u]:
            test_file.write(str(u) + '\t')
            test_file.write(str(current_poi) + '\t')
            test_file.write(str(current_time) + '\t')
            for p in seq_set:
                test_file.write(str(p) + ' ')
            test_file.write('\t')
            test_file.write(str(ans_poi))
            test_file.write('\t')
            for p in ans_set:
                test_file.write(str(p) + ' ')
            test_file.write('\n')
    test_file.close()



def tuneing_data_genfile(filepath, dataset, time_threshold, seq_length, load_seq):
    train_list, tune_list, ans, tune_ans = dataset_pair2(filepath, dataset, 0.7, 0.1, 3, time_threshold, seq_length, load_seq)
    tune_file = open(filepath + dataset + '_Tune2_' + str(time_threshold) + '.txt', 'w')
    for u in tune_ans.keys():
        current_poi, current_time, seq_set, ans_poi, ans_set = tune_ans[u]
        tune_file.write(str(u) + '\t')
        tune_file.write(str(current_poi) + '\t')
        tune_file.write(str(current_time) + '\t')
        for p in seq_set:
            tune_file.write(str(p) + ' ')
        tune_file.write('\t')
        tune_file.write(str(ans_poi))
        tune_file.write('\t')
        for p in ans_set:
            tune_file.write(str(p) + ' ')
        tune_file.write('\n')
    tune_file.close()
    

def dataset_pair2_o(filepath, dataset, train_percent, tune_percent, query_limit, time_threshold):
    train_list, tune_list, test_list = dataset_splitting2(filepath, dataset, train_percent, tune_percent)
    training, tuning, testing = listtodict2(train_list, 1), listtodict2(tune_list, 0), listtodict2(test_list, 0)
    tune_dict = listtodict2(tune_list, 1)
    ans, tune_ans = defaultdict(list), defaultdict(list)
    for u in training.keys():
        visited = set(training[u].keys())
        if len(visited) == 0:
            continue
        #   Get Tuning Pair
        if u in tuning.keys():
            tuning_pair = []
            ans_set = set()
            seq_set = set()
            for i in range(len(tuning[u]) - 1):
                (p1, d1) = tuning[u][i]
                (p2, d2) = tuning[u][i + 1]
                if not ans_set:
                    start_time = d1
                    if p2 not in visited:
                        if timediff2(start_time, d2) / float(3600) <= time_threshold:
                            current_poi = p1
                            current_time = d1
                            seq_set = {p1}
                            ans_set |= {p2}
                            poi_ans = p2
                            for j in range(i - 1, -1, -1):
                                pj, dj = tuning[u][j]
                                if timediff2(dj, start_time) / float(3600) <= time_threshold:
                                    seq_set |= {pj}
                                else:
                                    break
                else:
                    if timediff2(start_time, d2) / float(3600) <= time_threshold:
                        if p2 not in visited:
                            ans_set |= {p2}
                            visited |= {p2}
                    else:
                        tuning_pair.append((current_poi, current_time, seq_set, poi_ans, ans_set))
                        ans_set = set()
                        seq_set = set()
            if len(tuning_pair) >= 1:
                tune_ans[u] = random.choice(tuning_pair)
            visited |= set(tune_dict[u].keys())

        #   Get Testing Pair
        if u in testing.keys():
            testing_pair = []
            ans_set = set()
            seq_set = set()
            for i in range(len(testing[u]) - 1):
                (p1, d1) = testing[u][i]
                (p2, d2) = testing[u][i + 1]
                if not ans_set:
                    start_time = d1
                    if p2 not in visited:
                        if timediff2(start_time, d2) / float(3600) <= time_threshold:
                            current_poi = p1
                            current_time = d1
                            seq_set = {p1}
                            ans_set |= {p2}
                            poi_ans = p2
                            for j in range(i - 1, -1, -1):
                                pj, dj = testing[u][j]
                                if timediff2(dj, start_time) / float(3600) <= time_threshold:
                                    seq_set |= {pj}
                                else:
                                    break
                else:
                    if timediff2(start_time, d2) / float(3600) <= time_threshold:
                        if p2 not in visited:
                            ans_set |= {p2}
                            visited |= {p2}
                    else:
                        testing_pair.append((current_poi, current_time, seq_set, poi_ans, ans_set))
                        ans_set = set()
                        seq_set = set()
            if len(testing_pair) >= query_limit:
                ans[u] = random.sample(testing_pair, query_limit)

    print('tune_case : ' + str(len(tune_ans.keys())))
    print('test_case : ' + str(len(ans.keys())))
    return train_list, tune_list, ans, tune_ans

def testing_data_genfile_o(filepath, dp, dataset, time_threshold):
    
    

    ufile = open(dp + dataset + '_userEncode_' + str(time_threshold) + '.txt', 'r')
    pfile = open(dp + dataset + '_poiEncode_' + str(time_threshold) + '.txt', 'r')
    
    ten_f = open(filepath + dataset + '_Testen_' + str(time_threshold) + '.txt', 'w')
    
    user_index = {}
    for line in ufile.readlines():
        info = line.strip().split('\t')
        user_index[info[0]] = info[1]
    
    ufile.close()
    
    poi_index = {}
    for line in pfile.readlines():
        info = line.strip().split('\t')
        poi_index[info[0]] = info[1]
    pfile.close()
    

    
    train_list, tune_list, ans, tune_ans = dataset_pair2_o(filepath, dataset, 0.7, 0.1, 3, time_threshold)
    
    test_file = open(filepath + dataset + '_Test_' + str(time_threshold) + '.txt', 'w')
    for u in ans.keys():
        for current_poi, current_time, seq_set, ans_poi, ans_set in ans[u]:

            if ans_poi == current_poi:
                judge = False
                ans_set = list(ans_set)
                for i in range(len(ans_set)):
                    if ans_set[i] != current_poi:
                        ans_poi = ans_set[i]
                        ans_set = ans_set[i+1:]
                        judge = False
                        break
                    else:
                        judge = True
                
                if judge == True:
                    continue
            test_file.write(str(u) + '\t')
            ten_f.write(str(user_index[str(u)]) + '\t')
            
            test_file.write(str(current_poi) + '\t')
            ten_f.write(str(poi_index[str(current_poi)]) + '\t')
            
            test_file.write(str(current_time) + '\t')
            ten_f.write(str(current_time) + '\t')
            
            for p in seq_set:
                test_file.write(str(p) + ' ')
                ten_f.write(str(poi_index[str(p)]) + ' ')
                
            test_file.write('\t')
            ten_f.write('\t')
            test_file.write(str(ans_poi))
            ten_f.write(str(poi_index[str(ans_poi)]))
            test_file.write('\t')
            ten_f.write('\t')
            for p in ans_set:
                test_file.write(str(p) + ' ')
                ten_f.write(str(poi_index[str(p)]) + ' ')
            test_file.write('\n')
            ten_f.write('\n')
    test_file.close()
    ten_f.close()
    
def tuning_data_genfile_o(filepath, dataset, time_threshold):
    train_list, tune_list, ans, tune_ans = dataset_pair2_o(filepath, dataset, 0.7, 0.1, 3, time_threshold)
    tune_file = open(filepath + dataset + '_Tune_' + str(time_threshold) + '.txt', 'w')
    for u in tune_ans.keys():
        current_poi, current_time, seq_set, ans_poi, ans_set = tune_ans[u]
        tune_file.write(str(u) + '\t')
        tune_file.write(str(current_poi) + '\t')
        for p in seq_set:
            tune_file.write(str(p) + ' ')
        tune_file.write('\t')
        tune_file.write(str(ans_poi))
        tune_file.write('\t')
        for p in ans_set:
            tune_file.write(str(p) + ' ')
        tune_file.write('\n')
    tune_file.close()

def poi_loader2(filename):
    geodict = {}
    geodata = open(filename, 'r')
    for line in geodata.readlines():
        info = line.strip().split('\t')
        # (id, lat, long)
        if len(info) == 3:
            geodict[int(info[0])] = (float(info[1]), float(info[2]))
    return geodict

def checkinstolist2(filename):
    

    clist = []
    ckdata = open(filename, 'r')
    for line in ckdata.readlines():
        info = line.strip().split('\t')
        """
        y = int(info[2][0:4])
        m = int(info[2][5:7])
        d = int(info[2][8:10])
        hh = int(info[2][11:13])
        mm = int(info[2][14:16])
        ss = int(info[2][17:19])
        clist.append((int(info[0]), int(info[1]), datetime.datetime(y, m, d, hh, mm, ss)))
        """
        
        y = int(info[2][0:4])
        m = int(info[2][5:7])
        d = int(info[2][8:10])
        hh = int(info[3][0:2])
        mm = int(info[3][3:5])
        ss = int(info[3][6:8])
        clist.append((int(info[0]), int(info[1]), datetime.datetime(y, m, d, hh, mm, ss)))
        
    ckdata.close()
    return clist


#   Checkinlist to Checkindict
def listtodict2(checkinlist, userorder):
    if userorder == 0:
        res = defaultdict(list)
    else:
        res = defaultdict(lambda: defaultdict(list))
    for checkin in checkinlist:
        if userorder == 0:
            res[checkin[0]].append((checkin[1], checkin[2]))
        elif userorder == 1:
            res[checkin[0]][checkin[1]].append(checkin[2])
        elif userorder == 2:
            res[checkin[1]][checkin[0]].append(checkin[2])
    return res



def socialgraph_loader2(filename):
    sgdata = open(filename, 'r')
    sdict = {}
    for line in sgdata.readlines():
        fl = line.strip().split('\t')
        usr = int(fl[0])
        for i in range(len(fl)):
            if i == 0:
                sdict[usr] = []
            else:
                sdict[usr].append(int(fl[i]))
    sgdata.close()
    return sdict


def checkinstolist3(filename):
    clist = []
    ckdata = open(filename, 'r')
    for line in ckdata.readlines():
        info = line.strip().split('\t')
        y = int(info[2][0:4])
        m = int(info[2][5:7])
        d = int(info[2][8:10])
        hh = int(info[2][11:13])
        mm = int(info[2][14:16])
        ss = int(info[2][17:19])
        clist.append((int(info[0]), int(info[1]), datetime.datetime(y, m, d, hh, mm, ss)))
    ckdata.close()
    return clist


def re_index(src_path, dataset, ts):
    checkinlist = checkinstolist(src_path + dataset + '_Checkins.txt')
    new_checkinlist = []
    user_poi_dict = listtodict(checkinlist, 1)
    geo_dict = poi_loader(src_path + dataset + '_PoiInfo.txt')
    
    user_index = {}
    poi_index = {}
    user, poi = 0, 0

    #   Construct Mapping
    for u in user_poi_dict.keys():
        user_index[u] = user
        user += 1
    for p in geo_dict.keys():
        poi_index[p] = poi
        poi += 1
    
   
    #   Update Checkins
    for u, p, d, t, c, lng, lat in checkinlist:
        new_checkinlist.append((user_index[u], poi_index[p], d, t, c, lng, lat))

    #   Generate Checkin File
    cfile = open(src_path + dataset + '_Checkins_re.txt', 'w')
    for c in new_checkinlist:
        cfile.write(str(c[0]) + '\t' + str(c[1]) + '\t' + str(c[2]) + '\t' + str(c[3]) + '\t' + str(c[4]) + '\t' + str(c[5]) + '\t' + str(c[6]) + '\n')
    cfile.close()

    #   Generate Poi Info
    pfile = open(src_path + dataset + '_PoiInfo_re.txt', 'w')
    for p in geo_dict.keys():
        pfile.write(str(poi_index[p]) + '\t' + str(geo_dict[p][0]) + '\t' + str(geo_dict[p][1]) + '\t' + str(geo_dict[p][2]) + '\n')
    pfile.close()
    
    
    test_query = testing_data_loading_o(src_path, dataset, ts)
    test_file = open(src_path + dataset + '_Test_re' + str(ts) + '.txt', 'w')
    for u in test_query.keys():
        for current_poi, (cur_day, cur_time), seq_set, ans_poi, ans_set in test_query[u]:
            test_file.write(str(user_index[u]) + '\t')
            test_file.write(str(poi_index[current_poi]) + '\t')
            test_file.write(str(cur_day) + ' ' + str(cur_time) + '\t')
            for p in seq_set:
                test_file.write(str(poi_index[p]) + ' ')
            test_file.write('\t')
            test_file.write(str(poi_index[ans_poi]))
            test_file.write('\t')
            for p in ans_set:
                test_file.write(str(poi_index[p]) + ' ')
            test_file.write('\n')
    test_file.close()


def re_index2(src_path, dst_path, dataset):
    
    checkinlist = checkinstolist3(src_path + dataset + '_Checkins.txt')
    new_checkinlist = []
    user_poi_dict = listtodict2(checkinlist, 1)
    geo_dict = poi_loader2(src_path + dataset + '_PoiInfo.txt')
    sdict = socialgraph_loader2(src_path + dataset + '_SortedGraph.txt')
    user_index = {}
    poi_index = {}
    user, poi = 0, 0

    #   Construct Mapping
    for u in user_poi_dict.keys():
        user_index[u] = user
        user += 1
    for p in geo_dict.keys():
        poi_index[p] = poi
        poi += 1

    #   Update Checkins
    for u, p, d in checkinlist:
        new_checkinlist.append((user_index[u], poi_index[p], d))

    #   Generate Checkin File
    cfile = open(dst_path + dataset + '_Checkins.txt', 'w')
    for c in new_checkinlist:
        cfile.write(str(c[0]) + '\t' + str(c[1]) + '\t' + str(c[2]) + '\n')
    cfile.close()

    #   Generate Poi Info
    pfile = open(dst_path + dataset + '_PoiInfo.txt', 'w')
    for p in geo_dict.keys():
        pfile.write(str(poi_index[p]) + '\t' + str(geo_dict[p][0]) + '\t' + str(geo_dict[p][1]) + '\n')
    pfile.close()

    #   Generate Social File
    sfile = open(dst_path + dataset + '_SortedGraph.txt', 'w')
    for u in sdict.keys():
        if u in user_index.keys():
            sfile.write(str(user_index[u]))
            for f in sdict[u]:
                if f in user_index.keys():
                    sfile.write('\t' + str(user_index[f]))
            sfile.write('\n')
    sfile.close()

    return user_index, poi_index

def fixdata(fp,ds):
    
    raw_file = open(fp + ds + '_Checkins2' + '.txt', 'r')
    write_file = open(fp + ds + '_Checkins'+ '.txt', 'w')
    for line in raw_file.readlines():
        info = line.strip().split('\t')
        write_file.write(str(info[0])+'\t'+str(info[1])+'\t'+str(info[2])+' '+str(info[3])+'\n')
         
    raw_file.close()
    write_file.close()
    
def del_time(fp,dp,ds,ts):
    
    raw_file = open(fp + ds + '_Test2_' + str(ts) + '.txt', 'r')
    write_file = open(dp + ds + '_Test_'+ str(ts) + '.txt', 'w')
    
    for line in raw_file.readlines():
        info = line.strip().split('\t')
        write_file.write(str(info[0])+'\t'+str(info[1])+'\t'+str(info[3])+'\t'+str(info[4])+'\t'+str(info[5])+'\n')
         
    raw_file.close()
    write_file.close()
    
    raw_file = open(fp + ds + '_Tune2_' + str(ts) + '.txt', 'r')
    write_file = open(dp + ds + '_Tune_'+ str(ts) + '.txt', 'w')
    
    for line in raw_file.readlines():
        info = line.strip().split('\t')
        write_file.write(str(info[0])+'\t'+str(info[1])+'\t'+str(info[3])+'\t'+str(info[4])+'\t'+str(info[5])+'\n')
         
    raw_file.close()
    write_file.close()

def load_user_vector(filepath,dataset):

    temp_dict = {}
    raw_file = open(filepath + dataset + '_Users_vector' + '.txt', 'r')
    for line in raw_file.readlines():
        info = line.strip().split('\t')
        temp = []
        for i in range(1,len(info)):
            temp.append(float(info[i]))
        myarray = np.asarray(temp)
        temp_dict[info[0]] = myarray
    raw_file.close()
    
    return temp_dict


def load_vector_poi(filepath,dataset):
    

    temp_dict = {}
    raw_file = open(filepath + dataset + '_graph2vec_Pois_vector' + '.txt', 'r')
    for line in raw_file.readlines():
        info = line.strip().split('\t')
        temp = []
        for i in range(1,len(info)):
            temp.append(float(info[i]))
        myarray = np.asarray(temp)
        temp_dict[info[0]] = myarray
    raw_file.close()
    
    return temp_dict

def load_vector_user(filepath,dataset):
    

    temp_dict = {}
    raw_file = open(filepath + dataset + '_graph2vec_Users_vector' + '.txt', 'r')
    for line in raw_file.readlines():
        info = line.strip().split('\t')
        temp = []
        for i in range(1,len(info)):
            temp.append(float(info[i]))
        myarray = np.asarray(temp)
        temp_dict[info[0]] = myarray
    raw_file.close()
    
    return temp_dict

def cal_normalize(d, target=1.0):
    raw = sum(d.values())
    try:
        factor = target/raw
    except ZeroDivisionError:
        factor = float(1)
    return {key:value*factor for key,value in d.items()}

def precision_score(y_true, y_pred):
    
    
    tp = list(set(y_true).intersection(set(y_pred)))  
    if len(tp) == 0:
        return 0
    else:
        return len(tp)/float(len(y_pred))

def recall_score(y_true, y_pred):
    
    tp = list(set(y_true).intersection(set(y_pred)))
    if len(tp) == 0:
        return 0
    else:
        return len(tp)/float(len(y_true))



def dist(a, b):    
    dis = geo_distance(a[0,0],a[1,0],b[0,0],b[1,0])
    return dis


def eps_neighbor(a, b, eps):
    
    return dist(a, b) < eps

def region_query(data, pointId, eps):
  
    nPoints = data.shape[1]
    seeds = []
    for i in range(nPoints):
        if eps_neighbor(data[:, pointId], data[:, i], eps):
            seeds.append(i)
    return seeds


def expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
   
    seeds = region_query(data, pointId, eps)
    if len(seeds) < minPts:
        clusterResult[pointId] = NOISE
        return False
    else:
        clusterResult[pointId] = clusterId 
        for seedId in seeds:
            clusterResult[seedId] = clusterId

        while len(seeds) > 0:
            currentPoint = seeds[0]
            queryResults = region_query(data, currentPoint, eps)
            if len(queryResults) >= minPts:
                for i in range(len(queryResults)):
                    resultPoint = queryResults[i]
                    if clusterResult[resultPoint] == UNCLASSIFIED:
                        seeds.append(resultPoint)
                        clusterResult[resultPoint] = clusterId
                    elif clusterResult[resultPoint] == NOISE:
                        clusterResult[resultPoint] = clusterId
            seeds = seeds[1:]
        return True


def dbscan(data, eps, minPts):

    clusterId = 1
    nPoints = data.shape[1]
    clusterResult = [UNCLASSIFIED] * nPoints
    for pointId in range(nPoints):
        point = data[:, pointId]
        if clusterResult[pointId] == UNCLASSIFIED:
            if expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
                clusterId = clusterId + 1
    return clusterResult, clusterId - 1




def all_poi_checked(datalist):
    X = []
    Y = []
    for i in datalist:
        X.append([float(i[5]),float(i[6])])
        Y.append(i[1])
    
    return X,Y

def gen_poi_categorydict(catelist,geo_dict):
    
    temp_dict = {}
    for i in catelist:
        if i[0] not in temp_dict:
            temp_dict[i[0]] = []
            
        if i[1] not in temp_dict:
            temp_dict[i[1]] = []
    
    for p in geo_dict:
        temp_dict[geo_dict[p][0]].append(p)

   

    return temp_dict

def gen_friends(fp,ds,datalist):
    

    raw_file = csv.reader(open(fp + ds + '_friendship_raw' + '.csv'))
    write_file = open(fp + ds + 'friendship'+ '.txt', 'w')
    
    alluser_checked_freq = user_checked(datalist)
    
    user = {}
    for title in raw_file:
        if title[0] not in user:
            user[title[0]] = []
        user[title[0]].append(title[1])
  
    effic_user = alluser_checked_freq.keys()
    
    friends_dict = {}
    keys = user.keys()
    key = sorted(keys)

    for i in key:
        if i in effic_user:
            write_file.write(str(i))
            friends_dict[i] = []
            for f in user[i]:
                if f in effic_user:
                    write_file.write('\t'+str(f))
                    friends_dict[i].append(f)
            write_file.write('\n')
        
    write_file.close()
    #raw_file.close()
    return friends_dict

class HITSIterator:


    def __init__(self, dg):
        self.max_iterations = 100  
        self.min_delta = 0.0001  
        self.graph = dg

        self.hub = {}
        self.authority = {}
        for node in self.graph.nodes():
            self.hub[node] = 1
            self.authority[node] = 1

    def hits(self):

        if not self.graph:
            return

        flag = False
        for i in range(self.max_iterations):
            change = 0.0  
            norm = 0  
            tmp = {}
    
            tmp = self.authority.copy()
            for node in self.graph.nodes():
                self.authority[node] = 0
                for ed in self.graph.edges():
                    if ed[1] == node:  
                        self.authority[node] += self.hub[ed[0]]
                norm += math.pow(self.authority[node], 2)

            norm = math.sqrt(norm)
            for node in self.graph.nodes():
                self.authority[node] /= norm
                change += abs(tmp[node] - self.authority[node])


            norm = 0
            tmp = self.hub.copy()
            for node in self.graph.nodes():
                self.hub[node] = 0
                for ed in self.graph.edges():
                    if ed[0] == node: 
                        self.hub[node] += self.authority[ed[1]]
                norm += math.pow(self.hub[node], 2)

            norm = math.sqrt(norm)
            for node in self.graph.nodes():
                self.hub[node] /= norm
                change += abs(tmp[node] - self.hub[node])

            print("This is NO.%s iteration" % (i + 1))
            print("authority", self.authority)
            print("hub", self.hub)

            if change < self.min_delta:
                flag = True
                break
        
        if flag:
            print("finished in %s iterations!" % (i + 1))
        else:
            print("finished out of 100 iterations!")
            
        print("The best authority page: ", max(self.authority.items(), key=lambda x: x[1]))
        print("The best hub page: ", max(self.hub.items(), key=lambda x: x[1]))
        
        temp = {}
        for p in self.authority:
            temp[p] = self.authority[p]*self.hub[p]  
        return temp  
        

def gen_time_successive(data_list,ts):
    
    nosuccessive_dict = {}
    successive_dict = {}
    user_record = listtodict(data_list, 0)
    for u in user_record.keys():

        if len(user_record[u]) == 1:
            if u not in nosuccessive_dict:
                nosuccessive_dict[u] = []
            (p1, d1, t1) = user_record[u][0]
            nosuccessive_dict[u].append((p1,d1,t1))
            
        
        if len(user_record[u]) > 1:  
            for q in range(len(user_record[u])):
                (p1, d1, t1) = user_record[u][q]  

                j = q+1  
                after = []  
                
                for k in range(j, len(user_record[u])):
                    (p2, d2, t2) = user_record[u][k]
                    if abs(timediff(d1, t1, d2, t2)) / float(3600) <= ts:
                        after.append(p2)
                    else:
                        break
                
                
                j = q
                before = []
                
                for k in range(0, j):
                    (p0, d0, t0) = user_record[u][k]
                    if abs(timediff(d0, t0, d1, t1)) / float(3600) <= ts:
                        before.append((p0,d0,t0))
                    else:
                        break
                
                if len(after) == 0 and len(before) == 0:   
                    if u not in nosuccessive_dict:
                        nosuccessive_dict[u] = []
                    nosuccessive_dict[u].append((p1,d1,t1))
                else:
                    if u not in successive_dict:
                        successive_dict[u] = []
                    successive_dict[u].append((before,p1,after))
    return nosuccessive_dict, successive_dict

def gen_successive_behavior(data_list,ts):

    nosuccessive_dict = {}
    successive_dict = {}
    user_record = listtodict(data_list, 0)
    for u in user_record.keys():

        if len(user_record[u]) == 1:
            if u not in nosuccessive_dict:
                nosuccessive_dict[u] = []
            (p1, d1, t1) = user_record[u][0]
            nosuccessive_dict[u].append(p1)
            
        
        if len(user_record[u]) > 1:  
            for q in range(len(user_record[u])):
                (p1, d1, t1) = user_record[u][q]  

                j = q+1  
                after = []  
                
                for k in range(j, len(user_record[u])):
                    (p2, d2, t2) = user_record[u][k]
                    if abs(timediff(d1, t1, d2, t2)) / float(3600) <= ts:
                        after.append(p2)
                    else:
                        break
                
                
                j = q
                before = []
                
                for k in range(0, j):
                    (p0, d0, t0) = user_record[u][k]
                    if abs(timediff(d0, t0, d1, t1)) / float(3600) <= ts:
                        before.append(p0)
                    else:
                        break
                
                if len(after) == 0 and len(before) == 0:   
                    if u not in nosuccessive_dict:
                        nosuccessive_dict[u] = []
                    nosuccessive_dict[u].append(p1)
                else:
                    if u not in successive_dict:
                        successive_dict[u] = []
                    successive_dict[u].append((before,p1,after))
    
    return nosuccessive_dict, successive_dict

def gen_testing(tr_list, te_list, successive_time_constrain, alluser_checked_freq):
    # user visited in train list
    user_visit = {}
    for user in list(alluser_checked_freq.keys()):
        user_visit[user] = set()
    for uid, lid, _, _ in tr_list:
        user_visit[uid].add(lid)
        

    nosuccessive_dict = {}
    successive_dict = {}
    user_record = listtodict(te_list, 0)
    for u in user_record.keys():

        if len(user_record[u]) == 1:
            if u not in nosuccessive_dict:
                nosuccessive_dict[u] = []
            (p1, d1, t1) = user_record[u][0]
            nosuccessive_dict[u].append(p1)

        if len(user_record[u]) > 1:
            for q in range(len(user_record[u])):
            
                (p1, d1, t1) = user_record[u][q]  

                j = q+1  
                after = []  
                
                for k in range(j, len(user_record[u])):
                    (p2, d2, t2) = user_record[u][k]
                    if abs(timediff(d1, t1, d2, t2)) / float(3600) <= successive_time_constrain:
                        after.append(p2)
                    else:
                        break
                
                
                j = q
                before = []
                
                for k in range(j-1, -1, -1):
                    (p0, d0, t0) = user_record[u][k]
                    if abs(timediff(d0, t0, d1, t1)) / float(3600) <= successive_time_constrain:
                        before.insert(0, p0)
                    else:
                        break

                """

                for k in range(0, j):
                    (p0, d0, t0) = user_record[u][k]
                    if abs(timediff(d0, t0, d1, t1)) / float(3600) <= successive_time_constrain:
                        before.append(p0)
                    else:
                        break
                """

                if len(after) == 0 and len(before) == 0:   
                    if u not in nosuccessive_dict:
                        nosuccessive_dict[u] = []
                    nosuccessive_dict[u].append(p1)
                else:
                    gd = set(after) - set(user_visit[u])
                    if len(gd) == 0:
                        continue
                    if u not in successive_dict:
                        successive_dict[u] = []
                    successive_dict[u].append((before,p1,list(gd)))
                    
                    
                    #user_visit[u] |= set(after)
    
    return successive_dict
 
def gen_sequence_split_by_exist_context(data_list,ts):

    nosuccessive_dict = {}
    successive_dict = {}
    user_record = listtodict(data_list, 0)
    for u in user_record.keys():

        if len(user_record[u]) == 1:
            if u not in nosuccessive_dict:
                nosuccessive_dict[u] = []
            (p1, d1, t1) = user_record[u][0]
            nosuccessive_dict[u].append(([],p1,[]))
            
        
        if len(user_record[u]) > 1:  
            for q in range(len(user_record[u])):
                (p1, d1, t1) = user_record[u][q]  

                j = q+1  
                after = []  
                
                for k in range(j, len(user_record[u])):
                    (p2, d2, t2) = user_record[u][k]
                    if abs(timediff(d1, t1, d2, t2)) / float(3600) <= ts:
                        after.append(p2)
                    else:
                        break
                
                
                j = q
                before = []
                
                for k in range(j-1, -1, -1):
                    (p0, d0, t0) = user_record[u][k]
                    if abs(timediff(d0, t0, d1, t1)) / float(3600) <= ts:
                        before.insert(0, p0)
                    else:
                        break

                """

                for k in range(0, j):
                    (p0, d0, t0) = user_record[u][k]
                    if abs(timediff(d0, t0, d1, t1)) / float(3600) <= ts:
                        before.append(p0)
                    else:
                        break
                """
                if len(before) == 0:  # no context
                    if u not in nosuccessive_dict:
                        nosuccessive_dict[u] = []
                    nosuccessive_dict[u].append((before,p1,after))
                else:
                    if u not in successive_dict:
                        successive_dict[u] = []
                    successive_dict[u].append((before,p1,after))
    
    return nosuccessive_dict, successive_dict
            
def save_training_sequence(filepath, dataset, ts, file_type, data_list):

    nosuccessive_dict = {}
    successive_dict = {}
    user_record = listtodict(data_list, 0)
    for u in user_record.keys():

        if len(user_record[u]) == 1:
            if u not in nosuccessive_dict:
                nosuccessive_dict[u] = []
            (p1, d1, t1) = user_record[u][0]
            nosuccessive_dict[u].append(([],p1,[]))


        if len(user_record[u]) > 1:
            for q in range(len(user_record[u])):
                (p1, d1, t1) = user_record[u][q]

                j = q+1
                after = []

                for k in range(j, len(user_record[u])):
                    (p2, d2, t2) = user_record[u][k]
                    if abs(timediff(d1, t1, d2, t2)) / float(3600) <= ts:
                        after.append(p2)
                    else:
                        break

                j = q
                before = []

                for k in range(j-1, -1, -1):
                    (p0, d0, t0) = user_record[u][k]
                    if abs(timediff(d0, t0, d1, t1)) / float(3600) <= ts:
                        before.insert(0, p0)
                    else:
                        break

                """
                for k in range(0, j):
                    (p0, d0, t0) = user_record[u][k]
                    if abs(timediff(d0, t0, d1, t1)) / float(3600) <= ts:
                        before.append(p0)
                    else:
                        continue
                """

                if len(after) == 0:  # no future
                    if u not in nosuccessive_dict:
                        nosuccessive_dict[u] = []
                    nosuccessive_dict[u].append((before,p1,after))
                else:
                    if u not in successive_dict:
                        successive_dict[u] = []
                    cktime = datetime.datetime.strptime(d1+" "+t1, "%Y-%m-%d %H:%M:%S")
                    successive_dict[u].append((before,p1,after,cktime))

    training_seq = []
    for usr in successive_dict:
        for item in successive_dict[usr]:
            bef, cur, aft, ckt = item
            bt = bef.copy()
            bt.append(cur)
            context = bt
            if len(context) > 3:
                context = context[-3:]
            while len(context) < 3:
                context = ["0"] + context
            con = np.array(context)
            training_seq.append((usr, ckt, con, aft))

    with open(filepath + dataset + '_' + file_type+ '_' + str(ts) + '.txt', 'w') as opt:
        for usr, ckt, con, fut in training_seq:
            opt.write(usr)
            opt.write("\t")
            opt.write(' '.join(con))
            opt.write("\t")
            opt.write(ckt.strftime("%Y-%m-%d %H:%M:%S"))
            opt.write("\t")
            opt.write(' '.join(fut))
            opt.write("\n")
    return training_seq

def save_tune_or_test(filepath, dataset, ts, file_type, tr_list, te_list, alluser_checked_freq):
    user_visit = {}
    for user in list(alluser_checked_freq.keys()):
        user_visit[user] = set()
    for uid, lid, _, _ in tr_list:
        user_visit[uid].add(lid)

    nosuccessive_dict = {}
    successive_dict = {}
    user_record = listtodict(te_list, 0)
    for u in user_record.keys():

        if len(user_record[u]) == 1:
            if u not in nosuccessive_dict:
                nosuccessive_dict[u] = []
            (p1, d1, t1) = user_record[u][0]
            nosuccessive_dict[u].append(([],p1,[]))


        if len(user_record[u]) > 1:
            for q in range(len(user_record[u])):
                (p1, d1, t1) = user_record[u][q]

                j = q+1
                after = []

                for k in range(j, len(user_record[u])):
                    (p2, d2, t2) = user_record[u][k]
                    if abs(timediff(d1, t1, d2, t2)) / float(3600) <= ts:
                        after.append(p2)
                    else:
                        break

                j = q
                before = []

                for k in range(j-1, -1, -1):
                    (p0, d0, t0) = user_record[u][k]
                    if abs(timediff(d0, t0, d1, t1)) / float(3600) <= ts:
                        before.insert(0, p0)
                    else:
                        break
                """
                for k in range(0, j):
                    (p0, d0, t0) = user_record[u][k]
                    if abs(timediff(d0, t0, d1, t1)) / float(3600) <= ts:
                        before.append(p0)
                    else:
                        continue
                """

                if len(after) == 0:  # no future
                    if u not in nosuccessive_dict:
                        nosuccessive_dict[u] = []
                    nosuccessive_dict[u].append((before,p1,after))
                else:
                    gd = set(after) - set(user_visit[u])
                    if len(gd) == 0:
                        continue
                    if u not in successive_dict:
                        successive_dict[u] = []
                    cktime = datetime.datetime.strptime(d1+" "+t1, "%Y-%m-%d %H:%M:%S")
                    successive_dict[u].append((before,p1,list(gd),cktime))

    training_seq = []
    for usr in successive_dict:
        for item in successive_dict[usr]:
            bef, cur, aft, ckt = item
            bt = bef.copy()
            bt.append(cur)
            context = bt
            if len(context) > 3:
                context = context[-3:]
            while len(context) < 3:
                context = ["0"] + context
            con = np.array(context)
            training_seq.append((usr, ckt, con, aft))

    with open(filepath + dataset + '_' + file_type+ '_' + str(ts) + '.txt', 'w') as opt:
        for usr, ckt, con, fut in training_seq:
            opt.write(usr)
            opt.write("\t")
            opt.write(' '.join(con))
            opt.write("\t")
            opt.write(ckt.strftime("%Y-%m-%d %H:%M:%S"))
            opt.write("\t")
            opt.write(' '.join(fut))
            opt.write("\n")
    return training_seq

def gen_behavior(data_list,ts):   
    
    
    nosuccessive_dict = {}
    successive_dict = {}
    user_record = listtodict(data_list, 0)
    for u in user_record.keys():

        if len(user_record[u]) == 1:
            if u not in nosuccessive_dict:
                nosuccessive_dict[u] = []
            (p1, d1, t1) = user_record[u][0]
            nosuccessive_dict[u].append((p1,d1,t1))
        
        i = 0
        if len(user_record[u])>1:
            while i<len(user_record[u]):
                (p1, d1, t1) = user_record[u][i]
                poiset = [] 
                j = i+1
                while j<len(user_record[u]):
                    (p2, d2, t2) = user_record[u][j]
                    if abs(timediff(d1, t1, d2, t2) / float(3600)) <= ts:
                        poiset.append((p2,d2,t2))
                        j = j+1
                        i = j
                    else:
                        i = j
                        break
                        
                if len(poiset) == 0:
                    if u not in nosuccessive_dict:
                        nosuccessive_dict[u] = []
                    nosuccessive_dict[u].append((p1,d1,t1))
                else:
                    if u not in successive_dict:
                        successive_dict[u] = []
                    temp = []
                    temp.append((p1,d1,t1))
                    for p in poiset:
                        temp.append(p)
                    successive_dict[u].append(temp)
                   
                if i == len(user_record[u])-1:
                    break
    
    return nosuccessive_dict, successive_dict


def gen_poi_raw(fp,ds):
    
    
    f1 = open(fp + ds + '_Checkins' + '.txt', 'r')
    f2 = open(fp + ds + '_PoiInfo_raw'+ '.txt', 'w')


    poichecked = {}

    try:
        while True:
            line = f1.readline()
            line=line.strip('\n')
            line=line.strip('\t')
            if not line:
                break
            record = line.split('\t')
            poichecked.setdefault(record[1],[]).append([record[4],record[5],record[6]])
       


    finally:
        print("first level End.....")
        ### rawdata: long,
        keys = poichecked.keys()
        keys = sorted(keys)
        for key in keys:
            f2.write(str(key)+'\t'+str(poichecked[key][0][0])+'\t'+str(poichecked[key][0][1])+'\t'+str(poichecked[key][0][2])+'\n')
    f1.close()
    f2.close()

class Node:
    def __init__(self,freq):
        self.left = None
        self.right = None
        self.father = None
        self.freq = freq
    def isLeft(self):
        return self.father.left == self

def createNodes(freqs):
    return [Node(freq) for freq in freqs]

def createHuffmanTree(nodes):
    queue = nodes[:]
    while len(queue) > 1:
        queue.sort(key=lambda item:item.freq)
        node_left = queue.pop(0)
        node_right = queue.pop(0)
        node_father = Node(node_left.freq + node_right.freq)
        node_father.left = node_left
        node_father.right = node_right
        node_left.father = node_father
        node_right.father = node_father
        queue.append(node_father)
    queue[0].father = None
    return queue[0]


def getlevel(rootnode):
    level = 0
    thislevel = [rootnode]
    while thislevel:
        nextlevel = list()
        for n in thislevel:
            if n.left: nextlevel.append(n.left)
            if n.right: nextlevel.append(n.right)
        thislevel = nextlevel
        level = level+1
    return level


def traverse_path(tree, target_poi, code):
    paths = []
    if not (tree.left or tree.right):
        return [[(tree.value,code)]]
    if tree.left:
        paths.extend([[(tree.value,'left')] + child for child in traverse_path(tree.left,target_poi,code+'0')])
    if tree.right:
        paths.extend([[(tree.value,'right')] + child for child in traverse_path(tree.right,target_poi,code+'1')])  
        
    return paths


def traverse3(rootnode):
    temp = []
    thislevel = [rootnode]
    while thislevel:
        nextlevel = list()
        for n in thislevel:
            temp.append(n.value)
            if n.left: nextlevel.append(n.left)
            if n.right: nextlevel.append(n.right)
        temp.append('#')
        thislevel = nextlevel
    return temp



def traverse2(rootnode):
    temp = []
    thislevel = [rootnode]
    while thislevel:
        nextlevel = list()
        for n in thislevel:
            temp.append(n.possibility)
            if n.left: nextlevel.append(n.left)
            if n.right: nextlevel.append(n.right)
        temp.append('#')
        thislevel = nextlevel
    return temp

def traverse(rootnode):
    temp = []
    thislevel = [rootnode]
    while thislevel:
        nextlevel = list()
        for n in thislevel:
            temp.append(n.val)
            if n.left: nextlevel.append(n.left)
            if n.right: nextlevel.append(n.right)
        temp.append('#')
        thislevel = nextlevel
    return temp



def huffman_encoding(nodes,root):
    codes = [''] * len(nodes)
    for i in range(len(nodes)):
        node_tmp = nodes[i]
        while node_tmp != root:
            if node_tmp.isLeft():
                codes[i] = '0' + codes[i]
            else:
                codes[i] = '1' + codes[i]
            node_tmp = node_tmp.father
    return codes


def geo_distance(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6372.797 * c 
    return km  



def sort_data(filepath,dataset):
    raw_file = open(filepath + dataset + '_filter' + '.txt', 'r')
    write_file = open(filepath + dataset + '_Checkins_sort'+ '.txt', 'w')
    time_dict = {}
    for line in raw_file.readlines():
        info = line.strip().split('\t')
        temp = info[2]+' '+info[3]
        d=datetime.datetime.strptime(temp,"%Y-%m-%d %H:%M:%S")
        sectime = time.mktime(d.timetuple())
        time_dict.setdefault(sectime,[]).append((info[0],info[1],info[2],info[3],info[4],info[5],info[6]))
    
    keys = time_dict.keys()
    keys = sorted(keys)
    for key in keys:
        for i in range(0,len(time_dict[key])):
            write_file.write(str(time_dict[key][i][0])+'\t'+str(time_dict[key][i][1])+'\t'+str(time_dict[key][i][2])+'\t'+str(time_dict[key][i][3])+'\t'+str(time_dict[key][i][4])+'\t'+str(time_dict[key][i][5])+'\t'+str(time_dict[key][i][6])+'\n')
         
    raw_file.close()
    write_file.close()

def checkin_loader(filepath, dataset):
    temp_dict = {}
    poi_dict = {}
    checkin_file = open(filepath + dataset + '_rawdata.txt', 'r')
    for line in checkin_file.readlines():
        info = line.strip().split('\t')
        temp_dict.setdefault(info[1],[]).append(info[0])
    checkin_file.close()
    for p in temp_dict:
        poi_dict[p] = len(temp_dict[p])
        
    return poi_dict


def checkinstolist(filename):
    clist = []
    ckdata = open(filename, 'r')
    for line in ckdata.readlines():
        info = line.strip().split('\t')
        clist.append((info[0], info[1],info[2],info[3]))
    ckdata.close()
    return clist

def userchecked_categorydict(filename):
    cdict = {}
    ckdata = open(filename, 'r')
    for line in ckdata.readlines():
        info = line.strip().split('\t')
        cdict.setdefault(info[0],[]).append([info[1],info[2],info[3],info[4]]) 
    ckdata.close()
    return cdict

def usercheckeddict(filename):
    cdict = {}
    ckdata = open(filename, 'r')
    for line in ckdata.readlines():
        info = line.strip().split('\t')
        cdict.setdefault(info[0],[]).append([info[1],info[2],info[3]]) 
    ckdata.close()
    return cdict

def poicheckeddict(filename):
    cdict = {}
    ckdata = open(filename, 'r')
    for line in ckdata.readlines():
        info = line.strip().split('\t')
        cdict.setdefault(info[1],[]).append([info[2],info[3]]) 
    ckdata.close()
    return cdict


def dataset_splitting(filepath, dataset, train_percent, tune_percent):
    checkinlist = checkinstolist(filepath + dataset + '_Checkins.txt')
    index_1 = int(round(len(checkinlist) * train_percent))
    index_2 = int(round(len(checkinlist) * (train_percent + tune_percent)))
    train = checkinlist[:index_1]
    tune = checkinlist[index_1:index_2]
    tests = checkinlist[index_2:]
    return train, tune, tests



def poi_loader(filename):
    geodict = {}
    geodata = open(filename, 'r')
    for line in geodata.readlines():
        info = line.strip().split('\t')
        if len(info) == 4:
            geodict[info[0]] = (info[1],float(info[2]), float(info[3]))
    return geodict


def timediff(date1,sec1, date2,sec2):
    
    temp1 = date1+' '+sec1
    d=datetime.datetime.strptime(temp1,"%Y-%m-%d %H:%M:%S")
    sectime = time.mktime(d.timetuple())
    
    temp2 = date2+' '+sec2
    d=datetime.datetime.strptime(temp2,"%Y-%m-%d %H:%M:%S")
    sectime2 = time.mktime(d.timetuple())
    
    return (sectime2-sectime)

def delrepeat(filepath,dataset):
    
    checkinlist = checkinstolist(filepath + dataset + '_Checkins_sort.txt')
    f = open(filepath + dataset + '_Checkins2' + '.txt', 'w')
    
    temp_line = list(set(checkinlist))
    
    time_dict = {}
    for line in temp_line:
        temp = line[2]+' '+line[3]
        d=datetime.datetime.strptime(temp,"%Y-%m-%d %H:%M:%S")
        sectime = time.mktime(d.timetuple())
        time_dict.setdefault(sectime,[]).append((line[0],line[1],line[2],line[3],line[4],line[5],line[6]))
    
    keys = time_dict.keys()
    keys = sorted(keys)
    for key in keys:
        for i in range(0,len(time_dict[key])):
            f.write(str(time_dict[key][i][0])+'\t'+str(time_dict[key][i][1])+'\t'+str(time_dict[key][i][2])+'\t'+str(time_dict[key][i][3])+'\t'+str(time_dict[key][i][4])+'\t'+str(time_dict[key][i][5])+'\t'+str(time_dict[key][i][6])+'\n')
         
    f.close()
    

def pois_count(data_list):
    temp_dict = {}
    pois_dict = {}
    for i in data_list:
        temp_dict.setdefault(i[1],[]).append(i[0])
    for p in temp_dict:
        pois_dict[p] = len(temp_dict[p])
    return pois_dict


def encode(poi_lng,poi_lat,precision):
    
    lng = [ -180.0 , 180.0]
    lat = [ -90.0, 90.0]
    code = ''
    num = 1
    while(num<=precision/2):

        mid_lng = (lng[0]+lng[1])/float(2)
        if poi_lng>mid_lng:
            lng[0] = mid_lng
            code = code+'1'
        else:
            lng[1] = mid_lng
            code = code+'0'
        
        mid_lat = (lat[0]+lat[1])/float(2)
        if poi_lat>mid_lat:
            lat[0] = mid_lat
            code = code+'1'
        else:
            lat[1] = mid_lat
            code = code+'0'
             
        num = num+1
   
    return code


def encode_cordinate(query_poi, head, filepath, dataset, precision):
    geo_dict = poi_loader(filepath + dataset + '_PoiInfo.txt')
    lng,lat = geo_dict[query_poi][1],geo_dict[query_poi][2]
    level = precision
    ptr = head
    code = ""
    for t in range(0,level):
        if t%2 == 0:  #  lng
            if lng <= ptr.possibility:
                code = code+'0'
                ptr = ptr.left
            else:
                code = code+'1'
                ptr = ptr.right
        else:
            if lat <= ptr.possibility:
                code = code+'0'
                ptr = ptr.left
            else:
                code = code+'1'
                ptr = ptr.right
            
    #print(query_poi,code)
    
def gen_grid_pois(query_poi,filepath, dataset,precision):
    
    #poislist = []
    geo_dict = poi_loader(filepath + dataset + '_PoiInfo.txt')
    rang = 180 / math.pi * 4.875 / float(6372.797)
    lngR = rang / math.cos(float(geo_dict[query_poi][2]) * math.pi / float(180));
    maxLng = float(geo_dict[query_poi][1]) + lngR
    minLng = float(geo_dict[query_poi][1]) - lngR
    maxLat = float(geo_dict[query_poi][2]) + rang
    minLat = float(geo_dict[query_poi][2]) - rang
    #for p in geo_dict:
        #if (geo_dict[p][1]>=minLng and geo_dict[p][1]<=maxLng) and (geo_dict[p][2]>=minLat and geo_dict[p][2]<=maxLat):
            #poislist.append(p)
    
    region_code = []
    code1 = encode(minLng,minLat,precision)
    code2 = encode(maxLng,minLat,precision)
    code3 = encode(minLng,maxLat,precision)
    code4 = encode(maxLng,maxLat,precision)
    region_code.append(code1[-2:])
    region_code.append(code2[-2:])
    region_code.append(code3[-2:])
    region_code.append(code4[-2:])
    region_code = list(set(region_code))
 
    
    
    return region_code

def gen_grid_pois2(query_poi,filepath, dataset,precision):
    
    geo_dict = poi_loader(filepath + dataset + '_PoiInfo.txt')
    rang = 180 / math.pi * 4.875 / float(6372.797)
    lngR = rang / math.cos(float(geo_dict[query_poi][2]) * math.pi / float(180));
    maxLng = float(geo_dict[query_poi][1]) + lngR
    minLng = float(geo_dict[query_poi][1]) - lngR
    maxLat = float(geo_dict[query_poi][2]) + rang;
    minLat = float(geo_dict[query_poi][2]) - rang;
   
    
    return minLng,maxLng,minLat,maxLat


def gen_person_huffman_pois(query_poi, fp, ds):
    
    
    geo_dict = poi_loader(fp + ds + '_PoiInfo.txt')
    rang = 180 / math.pi * 4.875/ float(6372.797)
    lngR = rang / math.cos(float(geo_dict[query_poi][2]) * math.pi / float(180));
    maxLng = float(geo_dict[query_poi][1]) + lngR
    minLng = float(geo_dict[query_poi][1]) - lngR
    maxLat = float(geo_dict[query_poi][2]) + rang;
    minLat = float(geo_dict[query_poi][2]) - rang;
    
    poiset = []
    for p in geo_dict:
        if (geo_dict[p][1]>=minLng and geo_dict[p][1]<=maxLng) and (geo_dict[p][2]>=minLat and geo_dict[p][2]<=maxLat):
            poiset.append(p)
    
    return poiset



def poi_encode_loader(filepath, dataset, precision):
    rg_dict = {}
    rg_file = open(filepath + dataset + '_Encode_' + str(precision) + '.txt', 'r')
    for line in rg_file.readlines():
        info = line.strip().split('\t')
        rg_dict[info[0]] = info[1]
    rg_file.close()
    return rg_dict

def gen_poi_encode(filepath, dataset, precision):
    
    geo_dict = poi_loader(filepath + dataset + '_PoiInfo.txt')
    rg_file = open(filepath + dataset + '_Encode_' + str(precision) + '.txt', 'w')
    for p in geo_dict:
        lng, lat = geo_dict[p][1],geo_dict[p][2]
        code = encode(lng, lat, precision)
        rg_file.write(str(p) + '\t' + str(code) + '\n')
    rg_file.close()


def listtodict(checkinlist, userorder):
    if userorder == 0:
        res = defaultdict(list)
    else:
        res = defaultdict(lambda: defaultdict(list))
    for checkin in checkinlist:
        if userorder == 0:
            res[checkin[0]].append((checkin[1], checkin[2],checkin[3]))
        elif userorder == 1:
            res[checkin[0]][checkin[1]].append((checkin[2],checkin[3]))
        elif userorder == 2:
            res[checkin[1]][checkin[0]].append((checkin[2],checkin[3]))
    return res


def listtogh_dict(gh_dict, checkinlist, userorder):
    if userorder == 0:
        res = defaultdict(list)
    else:
        res = defaultdict(lambda: defaultdict(list))
    for checkin in checkinlist:
        if userorder == 0:
            res[checkin[0]].append((gh_dict[checkin[1]], checkin[2],checkin[3]))
        elif userorder == 1:
            res[checkin[0]][checkin[1]].append((checkin[2],checkin[3]))
        elif userorder == 2:
            res[checkin[1]][checkin[0]].append((checkin[2],checkin[3]))
    return res

def poi_layer(train, time_threshold, poi_nodes):
    user_train = listtodict(train, 0)
    d = nx.DiGraph()
    d.add_nodes_from(poi_nodes)
    for u in user_train.keys():
        for i in range(len(user_train[u]) - 1):
            (p1, d1, t1) = user_train[u][i]
            (p2, d2, t2) = user_train[u][i+1]
            if abs(timediff(d1, t1, d2, t2)) / float(3600) <= time_threshold and p1 != p2:
                if d.has_edge(p1, p2):     
                    d[p1][p2]['weight'] += 1
                else:
                    d.add_edge(p1, p2, weight=1)
    return d

def grid_layer(gh_dict, train, time_threshold, poi_nodes):
    user_train = listtogh_dict(gh_dict, train, 0)

    d = nx.DiGraph()
    d.add_nodes_from(poi_nodes)
    for u in user_train.keys():
        for i in range(len(user_train[u]) - 1):
            (p1, d1, t1) = user_train[u][i]
            (p2, d2, t2) = user_train[u][i+1]
            if abs(timediff(d1, t1, d2, t2)) / float(3600) <= time_threshold:
                if d.has_edge(p1, p2):     
                    d[p1][p2]['weight'] += 1
                else:
                    d.add_edge(p1, p2, weight=1)
    return d


def gen_successive_query(data_list, time_threshold):
    
    successive_dict = {}
    user_record = listtodict(data_list, 0)
    for u in user_record.keys():
        if len(user_record[u])>1:
            temp_record = user_record[u]
            i = 0
            while i<len(temp_record)-1:
                (p1, d1, t1) = temp_record[i]
                j=i+1
                (p2, d2, t2) = temp_record[j]
                if p1 == p2:
                    temp_record = temp_record[:j]+temp_record[j+1:]
                else:
                    i = i+1
 
                if j == len(temp_record):
                    break
            if len(temp_record)!=0:
                for i in range(0,len(temp_record)):
                    prev_pois = []
                    next_pois = []
                    (p1, d1, t1) = temp_record[i]
                    temp_prev_pois = temp_record[:i]
                    temp_next_pois = temp_record[i+1:]
                    
                    if len(temp_prev_pois)!=0:
                        for pr in temp_prev_pois:
                            (p2, d2, t2) = pr
                            if abs(timediff(d2, t2, d1, t1)) / float(3600) <= time_threshold:
                                prev_pois.append(p2)
                    
                    prev_pois.append(p1)
                    
                  
                    if len(temp_next_pois)!=0:
                        for ne in temp_next_pois:
                            (p3, d3, t3) = ne
                            if abs(timediff(d1, t1, d3, t3)) / float(3600) <= time_threshold:
                                next_pois.append(p3)
                            else:
                                break
                    
                    
                    if len(next_pois)!=0 and len(prev_pois)<=3:
                        if u not in successive_dict:
                            successive_dict[u] = []
                        successive_dict[u].append([prev_pois,next_pois])
                    elif len(next_pois)!=0 and len(prev_pois)>3:
                        prev_pois = prev_pois[-3:]
                        if u not in successive_dict:
                            successive_dict[u] = []
                        successive_dict[u].append([prev_pois,next_pois])
                    
                    """
                    if len(next_pois)!=0:
                        if u not in successive_dict:
                            successive_dict[u] = [] 
                        successive_dict[u].append([prev_pois,next_pois])
                    """
                     
  
    
    return successive_dict



def gen_successive_queryoftime(data_list, time_threshold):
    
    successive_dict = {}
    user_record = listtodict(data_list, 0)
    for u in user_record.keys():
        if len(user_record[u])>1:
            temp_record = user_record[u]
            i = 0
            while i<len(temp_record)-1:
                (p1, d1, t1) = temp_record[i]
                j = i+1
                (p2, d2, t2) = temp_record[j]
                if p1 == p2:
                    temp_record = temp_record[:j]+temp_record[j+1:]
                else:
                    i = i+1
                    
                if j == len(temp_record):
                    break
            poiset = [] 
            i = 0   
            if len(temp_record)>1:
                while i<len(temp_record)-1:
                    (p1, d1, t1) = temp_record[i]
                    while i<len(temp_record)-1:
                        j = i+1
                        (p2, d2, t2) = temp_record[j]
                        if abs(timediff(d1, t1, d2, t2) / float(3600)) <= time_threshold:
                            poiset.append((p2,d2,t2))
                            i = i+1
                        else:
                            i = j
                            break
                    if len(poiset)!=0:
                        if u not in successive_dict:
                            successive_dict[u] = []
                        temp = []
                        temp.append((p1,d1,t1))
                        for p in poiset:
                            temp.append(p)
                        successive_dict[u].append(temp)
                        poiset = []                  
                    if i == len(temp_record)-1:
                        break
    return successive_dict


def user_checked(tr_list):
    
    user_checked_freq = {}
    for i in tr_list:
        if i[0] not in user_checked_freq:
            user_checked_freq[i[0]] = {}
        
        if i[1] not in user_checked_freq[i[0]]:
            user_checked_freq[i[0]][i[1]] = 1
        else:
            user_checked_freq[i[0]][i[1]] = user_checked_freq[i[0]][i[1]]+1

    return user_checked_freq



def gen_transition_query(data_list, time_threshold):
    
    user_transition_dict = {}
    user_checkins = listtodict(data_list, 0)
    for u in user_checkins.keys():
        transition = []
        if len(user_checkins[u])>1:
            for i in range(0,len(user_checkins[u])-1):
                j=i+1
                (p1, d1, t1) = user_checkins[u][i]
                (p2, d2, t2) = user_checkins[u][j]
                if abs(timediff(d1, t1, d2, t2)) / float(3600) <= time_threshold and p1!=p2:
                    transition.append([p1,p2])
            if u not in user_transition_dict and len(transition)!=0:
                user_transition_dict[u] = []
                user_transition_dict[u]= transition
            elif u in user_transition_dict and len(transition)!=0:
                user_transition_dict[u] = transition
    
    return user_transition_dict
    


def gen_candidate_pois(fp,ds,distance_threshold,src):
    
    
    geo_dict = poi_loader(fp + ds + '_PoiInfo.txt')
    rang = 180 / math.pi * (distance_threshold/float(2)) / float(6372.797)
    lngR = rang / math.cos(float(geo_dict[src][2]) * math.pi / float(180));
    maxLng = float(geo_dict[src][1]) + lngR
    minLng = float(geo_dict[src][1]) - lngR
    maxLat = float(geo_dict[src][2]) + rang;
    minLat = float(geo_dict[src][2]) - rang;
    
    poiset = []
    for p in geo_dict:
        if (geo_dict[p][1]>=minLng and geo_dict[p][1]<=maxLng) and (geo_dict[p][2]>=minLat and geo_dict[p][2]<=maxLat):
            poiset.append(p)
    
    return poiset
    

def gen_poi_subgraph(poi_graph, candidate_poi_list):
    
    gpg = poi_graph.subgraph(candidate_poi_list)
    return gpg

def poi_in_grid(candidate_pois, visited_poi):
    
    pois = []
    for i in candidate_pois:
        if i in visited_poi:
            pois.append(i)
    return pois


def gen_pois_info_o(datalist, fp, ds):
    
    temp = []
    geo_dict = poi_loader(fp + ds + '_PoiInfo_raw.txt')
    geo_file = open(fp + ds + '_PoiInfo' + '.txt', 'w')
    for i in datalist:
        cate, lng, lat = geo_dict[i[1]][0], geo_dict[i[1]][1],geo_dict[i[1]][2]
        if i[1] not in temp:
            geo_file.write(str(i[1]) + '\t' + str(lat) + '\t' + str(lng) + '\n')
            temp.append(i[1])
    geo_file.close()


def gen_pois_info(datalist, fp, ds):
    
    temp = []
    geo_dict = poi_loader(fp + ds + '_PoiInfo_raw.txt')
    geo_file = open(fp + ds + '_PoiInfo' + '.txt', 'w')
    for i in datalist:
        cate, lng, lat = geo_dict[i[1]][0], geo_dict[i[1]][1],geo_dict[i[1]][2]
        if i[1] not in temp:
            geo_file.write(str(i[1]) + '\t' + str(cate) + '\t' + str(lng) + '\t' + str(lat) + '\n')
            temp.append(i[1])
    geo_file.close()

    
    
def learn_poisson(user_checked_freq):
    
    frq = {}
    lmbda_dict = {}
    for u in user_checked_freq:
        temp = []
        for p in user_checked_freq[u]:
            temp.append(user_checked_freq[u][p])
        freq_counter = Counter(temp)
        temp_list = []
        for c in range(1,11):
            if c not in freq_counter:
                temp_list.append(0)
            else:
                temp_list.append(freq_counter[c])
        frq[u] = temp_list
        
    for u in frq:
        if sum(frq[u]) == 0:
            lmbda_dict[u] = 0
        else:
            lmbda_dict[u] = sum(x*y for x, y in zip(range(1,11), frq[u]))/sum(frq[u])
            
    return lmbda_dict

def willness(a, xmin, dis):
    
    will = a * pow(log1p(dis), -xmin)
    return will


def freq_willness(b, ymin, freq):
    
    will = b * pow(log1p(freq), ymin)
    return will

def learn_power_law(user_checkins, geo_dict):
    dis_data = []
    for u in user_checkins.keys():
        user_dist = []
        for i in range(len(user_checkins[u]) - 1):
            p1, p2 = user_checkins[u][i][0], user_checkins[u][i + 1][0]
            if p1!=p2:
                dis = geo_distance(geo_dict[p1][1], geo_dict[p1][2], geo_dict[p2][1], geo_dict[p2][2])
                if 15 > dis > 0.0:
                    user_dist.append(log1p(dis))
        try:
            dis_data += random.sample(user_dist, 10)
        except ValueError:
            dis_data += user_dist
    np.seterr(divide='ignore', invalid='ignore')
    pw = powerlaw.Fit(data=dis_data)
    print('PW Finish')

    return pw.power_law.alpha, pw.power_law.xmin
'''
def grid_learn_power_law(user_checkins, gh_dict):
    dis_data = []
    for u in user_checkins.keys():
        user_dist = []
        for i in range(len(user_checkins[u]) - 1):
            p1, p2 = user_checkins[u][i][0], user_checkins[u][i + 1][0]
            if p1!=p2:
                (lng1,lat1) = Gy.decode(gh_dict[p1])
                (lng2,lat2) = Gy.decode(gh_dict[p2])
                grid_dis = geo_distance(float(lng1),float(lat1),float(lng2),float(lat2))
                if 15 > grid_dis > 0.0:
                    user_dist.append(log1p(grid_dis))
        try:
            dis_data += random.sample(user_dist, 10)
        except ValueError:
            dis_data += user_dist
    np.seterr(divide='ignore', invalid='ignore')
    pw = powerlaw.Fit(data=dis_data)
    print('PW Finish')

    return pw.power_law.alpha, pw.power_law.xmin
'''


def gen_checked_detail(tr_list):
    
    user_check = {}
    poi_check = {}
    for i in tr_list:
        if i[1] not in poi_check:
            poi_check[i[1]] = []
        poi_check[i[1]].append(i[0])
        
        if i[0] not in user_check:
            user_check[i[0]] = []
        user_check[i[0]].append(i[1])
        
        
    for i in poi_check.keys():
        poi_check[i] = list(set(poi_check[i]))
        
    for i in user_check.keys():
        user_check[i] = list(set(user_check[i]))
    
    
    return poi_check, user_check
        


def gen_checked_grid(tr_list, gh_dict):
    
    user_check_grid = {}
    grid_check_user = {}
    for i in tr_list:
        if gh_dict[i[1]] not in grid_check_user:
            grid_check_user[gh_dict[i[1]]] = []
        grid_check_user[gh_dict[i[1]]].append(i[0])
        
        if i[0] not in user_check_grid:
            user_check_grid[i[0]] = []
        user_check_grid[i[0]].append(gh_dict[i[1]])
        
        
    for i in grid_check_user.keys():
        grid_check_user[i] = list(set(grid_check_user[i]))
        
    for i in user_check_grid.keys():
        user_check_grid[i] = list(set(user_check_grid[i]))
    
    
    return grid_check_user, user_check_grid    
    
def gen_pois_frequency(tr_list, total_pois):
    
    pois_freq = {}
    for i in tr_list:
        if i[1] not in pois_freq:
            pois_freq[i[1]] = 1
        else:
            pois_freq[i[1]] = pois_freq[i[1]]+1

    for j in total_pois:
        if j not in pois_freq:
            pois_freq[j] = 0
            
    return pois_freq


def gen_grids_frequency(tr_list, gh_dict):
    
    grids_freq = {}
    for i in tr_list:
        if gh_dict[i[1]] not in grids_freq:
            grids_freq[gh_dict[i[1]]] = 1
        else:
            grids_freq[gh_dict[i[1]]] = grids_freq[gh_dict[i[1]]]+1

    for j in gh_dict.keys():
        if j not in grids_freq:
            grids_freq[j] = 0
            
    return grids_freq



class POIWord2Vec():
    
    def __init__(self, poi_freq, graph_nodes, vec_len, learn_rate):
        
        self.vec_len = vec_len
        self.learn_rate = learn_rate
        self.the_poi_dict = None  
        self.huffman = None  
        self.gnerate_poi_dict(poi_freq, graph_nodes) 
    
    def gnerate_poi_dict(self, poi_freq, graph_nodes):

        poi_dict = {}
        if isinstance(poi_freq,dict):
            sum_count = sum(poi_freq.values())
            for poi in poi_freq:
                temp_dict = dict(poi = poi, freq = poi_freq[poi], possibility = 0 if poi_freq[poi] == 0 else poi_freq[poi]/float(sum_count), vector = graph_nodes[poi].node_vector, Huffman = None)
                poi_dict[poi] = temp_dict
        self.the_poi_dict = poi_dict

    def train_vector(self, poi_freq, graph_nodes, target_poi):  

        if self.huffman==None:
            self.huffman = PoisHuffmanTree(self.the_poi_dict, vec_len=self.vec_len)  
        
        self.Deal_Gram_CBOW(graph_nodes, target_poi)
        
    def Deal_Gram_CBOW(self, graph_nodes, target_poi):
        
        poi_huffman = self.the_poi_dict[target_poi]['Huffman']
        gram_vector = self.the_poi_dict[target_poi]['vector']
        e = self.GoAlong_Huffman(poi_huffman, gram_vector, self.huffman.root)
        self.the_poi_dict[target_poi]['vector'] += e
        self.the_poi_dict[target_poi]['vector'] = preprocessing.normalize(self.the_poi_dict[target_poi]['vector'])
        graph_nodes[target_poi].node_vector = self.the_poi_dict[target_poi]['vector']
        
    def GoAlong_Huffman(self, poi_huffman, input_vector, root):

        node = root
        e = np.zeros([1,self.vec_len])
        for level in range(poi_huffman.__len__()):
            huffman_charat = poi_huffman[level]
            q = self.__Sigmoid(input_vector.dot(node.value.T))
            grad = self.learn_rate * (1-int(huffman_charat)-q)
            e += grad * node.value
            node.value += grad * input_vector
            node.value = preprocessing.normalize(node.value)
            if huffman_charat=='0':
                node = node.left
            else:
                node = node.right
        return e

    def __Sigmoid(self,value):
        return 1/(1+math.exp(-value))
    
class PoisHuffmanTreeNode():
    
    def __init__(self,value,freq):
        
        self.possibility = freq
        self.left = None
        self.right = None
        self.value = value 
        self.Huffman = ""


class PoisHuffmanTree():
    
    def __init__(self, the_poi_dict, vec_len):
        self.vec_len = vec_len
        self.root = None
        
        if len(the_poi_dict)!=0:
            pois_dict_list = list(the_poi_dict.values())
            node_list = [PoisHuffmanTreeNode(x['poi'],x['possibility']) for x in pois_dict_list]
            self.build_tree(node_list)
            self.generate_huffman_code(self.root, the_poi_dict)
        
    def build_tree(self,node_list):
        while node_list.__len__()>1:
            i1 = 0  
            i2 = 1  
            if node_list[i2].possibility < node_list[i1].possibility :
                [i1,i2] = [i2,i1]
            for i in range(2,node_list.__len__()): 
                if node_list[i].possibility<node_list[i2].possibility :
                    i2 = i
                    if node_list[i2].possibility < node_list[i1].possibility :
                        [i1,i2] = [i2,i1]
            top_node = self.merge(node_list[i1],node_list[i2])
            if i1<i2:
                node_list.pop(i2)
                node_list.pop(i1)
            elif i1>i2:
                node_list.pop(i1)
                node_list.pop(i2)
            else:
                raise RuntimeError('i1 should not be equal to i2')
            node_list.insert(0,top_node)
        self.root = node_list[0]
        
    def generate_huffman_code(self, node, the_poi_dict):
        
        stack = [self.root]
        while (stack.__len__()>0):
            node = stack.pop()
    
            while node.left or node.right :
                code = node.Huffman
                node.left.Huffman = code + "0"
                node.right.Huffman = code + "1"
                stack.append(node.right)
                node = node.left
            poi = node.value
            code = node.Huffman

            the_poi_dict[poi]['Huffman'] = code

    def merge(self,node1,node2):
        top_pos = node1.possibility + node2.possibility
        top_node = PoisHuffmanTreeNode(np.zeros([1,self.vec_len]), top_pos)
        if node1.possibility > node2.possibility :
            top_node.left = node2
            top_node.right = node1
        else:
            top_node.left = node1
            top_node.right = node2
        return top_node



def train_pois_vector(q, query_pois, pois_freq_dict, graph_nodes, geo_dict, distance_threshold, vec_len, learn_rate , fp, ds):
    
    result = []
    for p in query_pois:
        temp_poi_freq = {}
        candidate_pois = gen_candidate_pois(fp, ds, distance_threshold, p)
        
        
        for i in candidate_pois:
            if i not in temp_poi_freq:
                temp_poi_freq[i] = 0
            temp_poi_freq[i] = pois_freq_dict[i]
    
        wv = POIWord2Vec(temp_poi_freq, graph_nodes, vec_len, learn_rate)
        wv.train_vector(temp_poi_freq, graph_nodes, p)
        result.append([p,graph_nodes[p].node_vector])
    q.put(result)
    
    
def minmax_norm(value_dict):
    if len(value_dict) == 0:
        return
    
    values = [value_dict[k] for k in value_dict.keys()]
    v_max = max(values)
    v_min = min(values)

    for k in value_dict.keys():
        try:

            if v_max - v_min == 0:
                value_dict[k] = float(1)
            else:
                value_dict[k] = (value_dict[k] - v_min) / float((v_max - v_min))

        except ZeroDivisionError:
            value_dict[k] = float(1) 


def minmax_norm2(value_dict):
    if len(value_dict) == 0:
        return
    
    values = [value_dict[k] for k in value_dict.keys()]
    v_max = max(values)
    v_min = min(values)

    for k in value_dict.keys():
        try:

            if v_max - v_min == 0:
                value_dict[k] = float(0)
            else:
                value_dict[k] = (value_dict[k] - v_min) / float((v_max - v_min))

        except ZeroDivisionError:
            value_dict[k] = float(0)

def category_layer(category_list, category_nodes):
    
    d = nx.DiGraph()
    d.add_nodes_from(category_nodes)
    for i in category_list:
        [src, dest] = i
        if d.has_edge(src, dest):     
            d[src][dest]['weight'] += 1
        else:
            d.add_edge(src, dest, weight=1)
    
    return d
            
    
def load_category_info(filepath, dataset): 
 
    category_file = open(filepath + dataset + '_Category.txt', 'r')
    category_list = []
    category_types = []
    for line in category_file.readlines():
        
        info = line.strip().split('\t')
        category_list.append([info[0],info[1]])
        if info[0] not in category_types:
            category_types.append(info[0])
        if info[1] not in category_types:
            category_types.append(info[1])
            
    category_file.close()
    
    return category_list, category_types

def goalong_category_Huffman(poi_huffman, input_vector, root, vec_len, learn_rate):
    
    
    node = root
    e = np.zeros([1,vec_len])
    for level in range(len(poi_huffman)):
        huffman_charat = poi_huffman[level]
        q = Sigmoid(np.dot(input_vector,node.value.T))
        grad = learn_rate * (int(huffman_charat)-q)
        e += grad * node.value
        node.value += (grad * input_vector)
        node.value = preprocessing.normalize(node.value)
        if huffman_charat=='0':
            node = node.left
        else:
            node = node.right
    return e


class ClusterNode():
    
    def __init__(self,name,pois_freq_dict,node_vector,vec_len,learn_rate):
        
        self.name = name
        self.vec_len = vec_len
        self.learn_rate = learn_rate
        self.node_vector = node_vector 
        self.edge_vector = None
        self.gnerate_poi_dict(name,pois_freq_dict)  

    def gnerate_poi_dict(self, target_categoty, pois_freq_dict):
    
        poi_dict = {}
        if isinstance(pois_freq_dict,dict):
            sum_count = sum(pois_freq_dict.values())
            for poi in pois_freq_dict:
                temp_dict = dict(poi = poi,freq = pois_freq_dict[poi],possibility = pois_freq_dict[poi]/float(sum_count) if sum_count!=0 else float(0),vector = np.random.random([1,self.vec_len]),Huffman = None)
                poi_dict[poi] = temp_dict
        self.poi_dict = poi_dict


def cluster_poi_train(target_cluster, cluser_dict, cluster_nodes, vec_len, lamda, learn_rate):
    
    for p in cluser_dict[target_cluster]:
        
        poi_huffman = cluster_nodes[target_cluster].poi_dict[p]['Huffman']
        node_vector = cluster_nodes[target_cluster].poi_dict[p]['vector']
        e = goalong_category_Huffman(poi_huffman,node_vector,cluster_nodes[target_cluster].huffman.root,vec_len,learn_rate)
        temp = e[0]
        temp = list(temp)
        temp = np.array(temp)
        node_vector += temp
        node_vector = preprocessing.normalize(node_vector)
        
        cluster_nodes[target_cluster].poi_dict[p]['vector'] = node_vector
        
class FreqNode():
    
    def __init__(self,pois_freq_dict,node_vector,vec_len,learn_rate):
        
        self.vec_len = vec_len
        self.learn_rate = learn_rate
        self.node_vector = node_vector 
        self.edge_vector = None
        self.gnerate_poi_dict(pois_freq_dict)  

    def gnerate_poi_dict(self, pois_freq_dict):
    
        poi_dict = {}
        if isinstance(pois_freq_dict,dict):
            sum_count = sum(pois_freq_dict.values())
            for poi in pois_freq_dict:
                temp_dict = dict(poi = poi,freq = pois_freq_dict[poi],possibility = pois_freq_dict[poi]/float(sum_count) if sum_count!=0 else float(0),vector = np.random.random([1,self.vec_len]),Huffman = None)
                poi_dict[poi] = temp_dict
        self.poi_dict = poi_dict

class CategoryNode():
    
    def __init__(self,name,pois_freq_dict,node_vector,vec_len,learn_rate):
        
        self.name = name
        self.vec_len = vec_len
        self.learn_rate = learn_rate
        self.node_vector = node_vector 
        self.edge_vector = None
        self.gnerate_poi_dict(name,pois_freq_dict)  

    def gnerate_poi_dict(self, target_categoty, pois_freq_dict):
    
        poi_dict = {}
        if isinstance(pois_freq_dict,dict):
            sum_count = sum(pois_freq_dict.values())
            for poi in pois_freq_dict:
                temp_dict = dict(poi = poi,freq = pois_freq_dict[poi],possibility = pois_freq_dict[poi]/float(sum_count) if sum_count!=0 else float(0),vector = np.random.random([1,self.vec_len]),Huffman = None)
                poi_dict[poi] = temp_dict
        self.poi_dict = poi_dict

class GraphNode():
    

    def __init__(self,name,pois_freq_dict,node_vector,vec_len,learn_rate):
        
        self.name = name
        self.vec_len = vec_len
        self.learn_rate = learn_rate
        self.node_vector = node_vector 
        self.edge_vector = None
        self.gnerate_poi_dict(name,pois_freq_dict)  

    def gnerate_poi_dict(self, target_categoty, pois_freq_dict):
    
        poi_dict = {}
        if isinstance(pois_freq_dict,dict):
            sum_count = sum(pois_freq_dict.values())
            for poi in pois_freq_dict:
                temp_dict = dict(poi = poi,freq = pois_freq_dict[poi],possibility = pois_freq_dict[poi]/float(sum_count) if sum_count!=0 else float(0),vector = np.random.random([1,self.vec_len]),Huffman = None)
                poi_dict[poi] = temp_dict
        self.poi_dict = poi_dict

class CategoryPOI2Vec():
    
    def __init__(self, poi_freq, graph_nodes, vec_len, learn_rate):
        
        self.vec_len = vec_len
        self.learn_rate = learn_rate
        self.the_poi_dict = None  
        self.huffman = None  
        self.gnerate_poi_dict(poi_freq, graph_nodes) 
    
    def gnerate_poi_dict(self, poi_freq, graph_nodes):

        poi_dict = {}
        if isinstance(poi_freq,dict):
            sum_count = sum(poi_freq.values())
            for poi in poi_freq:
                temp_dict = dict(poi = poi, freq = poi_freq[poi], possibility = 0 if poi_freq[poi] == 0 else poi_freq[poi]/float(sum_count), vector = graph_nodes[poi].node_vector, Huffman = None)
                poi_dict[poi] = temp_dict
        self.the_poi_dict = poi_dict

    def train_vector(self, graph_nodes, target_poi, train_set, vec_len, geo_dict, category_nodes, hier_cate, lamda, learn_rate, run_times):  

        if self.huffman==None:
            self.huffman = PoisHuffmanTree(self.the_poi_dict, vec_len=self.vec_len)  
            
        self.Deal_Gram_CBOW(graph_nodes, target_poi, train_set, vec_len, geo_dict, category_nodes, hier_cate, lamda, learn_rate, run_times)
        
        
        
    def Deal_Gram_CBOW(self, graph_nodes, target_poi, train_set, vec_len, geo_dict, category_nodes, hier_cate, lamda, learn_rate, run_times):
        
        poi_huffman = self.the_poi_dict[target_poi]['Huffman']
        
        gram_vector_sum = np.zeros([1,vec_len])
        for p in train_set:
            gram_vector_sum = gram_vector_sum+ graph_nodes[p].node_vector

        e = self.GoAlong_Huffman(poi_huffman, gram_vector_sum, self.huffman.root)
        
        
        for p in train_set:
            if p in self.the_poi_dict:
                self.the_poi_dict[p]['vector'] += e
                self.the_poi_dict[p]['vector'] = preprocessing.normalize(self.the_poi_dict[p]['vector'])
            graph_nodes[p].node_vector += e
            graph_nodes[p].node_vector = preprocessing.normalize(graph_nodes[p].node_vector)     
            train_category_vector(graph_nodes[p].node_vector,geo_dict[p][0],category_nodes,hier_cate,vec_len,lamda,learn_rate,run_times)
 
    def GoAlong_Huffman(self, poi_huffman, input_vector, root):

        node = root
        e = np.zeros([1,self.vec_len])
        for level in range(poi_huffman.__len__()):
            huffman_charat = poi_huffman[level]
            q = self.__Sigmoid(input_vector.dot(node.value.T))
            grad = self.learn_rate * (1-int(huffman_charat)-q)
            e += grad * node.value
            node.value += grad * input_vector
            node.value = preprocessing.normalize(node.value)
            if huffman_charat=='0':
                node = node.left
            else:
                node = node.right
        return e

    def __Sigmoid(self,value):
        return 1/(1+math.exp(-value))


def goalong_pois_Huffman(poi_huffman, input_vector, root, vec_len, learn_rate):
    
    
    node = root
    e = np.zeros([1,vec_len])
    for level in range(len(poi_huffman)):
        huffman_charat = poi_huffman[level]
        q = Sigmoid(np.dot(input_vector,node.value.T))
        grad = learn_rate * (int(huffman_charat)-q)
        e += grad * node.value
        node.value += (grad * input_vector)
        node.value = preprocessing.normalize(node.value)
        if huffman_charat=='0':
            node = node.left
        else:
            node = node.right
    return e

def goalong_user_Huffman(poi_huffman, input_vector, root, vec_len, learn_rate):
    
    
    node = root
    e = np.zeros([1,vec_len])
    for level in range(len(poi_huffman)):
        huffman_charat = poi_huffman[level]
        q = Sigmoid(np.dot(input_vector,node.value.T))
        grad = learn_rate * (int(huffman_charat)-q)
        e += grad * node.value
        
        if huffman_charat=='0':
            node = node.left
        else:
            node = node.right
    return e


def Sigmoid(value):
    return 1/(1+math.exp(-value))


class user_category_softmax:  

    def __init__(self, alfa, lamda,run_times):  
        self.alfa = alfa  
        self.lamda = lamda  
        self.run_times = run_times  
        
    def cal_e(self,x,l):
        
       
        theta_l = self.w[l]
        product = np.dot(theta_l,x)
        
        return math.exp(product)
    
    
    def cal_probability(self,x,j):
      

        molecule = self.cal_e(x,j)
        denominator = sum([self.cal_e(x,i) for i in self.k])
        if molecule == 0:
            return float(0)
        else:
            return molecule/float(denominator)
    
    
    def cal_partial_derivative(self,x,y,j):
        
        first = int(y==j)                          
        second = self.cal_probability(x,j)          
        return -x*(first-second) + self.lamda*self.w[j]
    
        
    def train(self, src, dest, category_nodes, node_vector, vec_len):
        
        self.k = category_nodes[src].edge_vector.keys() 
        self.w = category_nodes[src].edge_vector      
        time = 0
        
        derivatives = {}
        e = np.zeros([1,vec_len])
        while time < self.run_times:
            time += 1
                
            x = node_vector[0]
            y = dest
            x = list(x)
            x = np.array(x)
            
            for j in self.k:
                derivatives[j] = self.cal_partial_derivative(x,y,j)
                
            for j in self.k:
                if j == dest:
                    e = e+self.alfa * derivatives[j]
                #self.w[j] -= self.alfa * derivatives[j]
   
        return e


class category_softmax:  

    def __init__(self, alfa, lamda, run_times):  
        self.alfa = alfa  
        self.lamda = lamda  
        self.run_times = run_times  
        
    def cal_e(self,x,l):
        
       
        theta_l = self.w[l]
        product = np.dot(theta_l,x)
        
        return math.exp(product)
    
    
    def cal_probability(self,x,j):
      

        molecule = self.cal_e(x,j)
        denominator = sum([self.cal_e(x,i) for i in self.k])
        if molecule == 0:
            return float(0)
        else:
            return molecule/float(denominator)
    
    
    def cal_partial_derivative(self,x,y,j):
        
        first = int(y==j)                          
        second = self.cal_probability(x,j)          
        return -x*(first-second) + self.lamda*self.w[j]
    
        
    def train(self, src, dest, category_nodes, node_vector, vec_len):
        
        self.k = category_nodes[src].edge_vector.keys() 
        self.w = category_nodes[src].edge_vector      
        time = 0
        
        derivatives = {}
        e = np.zeros([1,vec_len])
        while time < self.run_times:
            time += 1
                
            x = node_vector[0]
            y = dest
            x = list(x)
            x = np.array(x)
            
            for j in self.k:
                derivatives[j] = self.cal_partial_derivative(x,y,j)
              
            for j in self.k:
                #if j == dest:
                    #e = e+self.alfa * derivatives[j]
                e+=self.alfa * derivatives[j]
                self.w[j] -= self.alfa * derivatives[j]
   
        return e

def user_category_goalong_Graph( src, dest, category_nodes, node_vector, vec_len, lamda, learn_rate, run_times):
    
    p = user_category_softmax(learn_rate,lamda,run_times)  
    e = p.train(src,dest, category_nodes, node_vector, vec_len)
    return e

def category_goalong_Graph( src, dest, category_nodes, node_vector, vec_len, lamda, learn_rate, run_times):
    
    p = category_softmax(learn_rate,lamda,run_times)  
    e = p.train(src,dest, category_nodes, node_vector, vec_len)
    return e


def category_poi_train(target_poi, node_vector, category_net, category_nodes, geo_dict, hier_cate, vec_len, lamda, learn_rate, run_times):
    
    target_category = geo_dict[target_poi][0]
    
    cate_path = []
    search = False
    if target_category not in hier_cate:
        for i in hier_cate:
            if target_category in hier_cate[i]:
                search = True
                cate_path.append(i)
                break
            else:
                for j in hier_cate[i]:
                    if target_category in hier_cate[i][j]:
                        cate_path.append(i)
                        cate_path.append(j)
                        search = True
                        break
        
            if search:
                break
   
    cate_path.append(target_category)

    cate_trpath = []
    for i in range(0,len(cate_path)-1):
        j=i+1
        cate_trpath.append([cate_path[i],cate_path[j]])
    

    for i in cate_trpath:
        src = i[0]
        dest = i[1]
        e = category_goalong_Graph(src,dest,category_nodes,node_vector,vec_len,lamda,learn_rate,run_times)
        temp = e[0]
        temp = list(temp)
        temp = np.array(temp)
        node_vector += temp
        node_vector = preprocessing.normalize(node_vector)
    

    poi_huffman = category_nodes[target_category].poi_dict[target_poi]['Huffman']
    e = goalong_pois_Huffman(poi_huffman,node_vector,category_nodes[target_category].huffman.root,vec_len,learn_rate)
    temp = e[0]
    temp = list(temp)
    temp = np.array(temp)
    node_vector += temp
    node_vector = preprocessing.normalize(node_vector)
    
    return node_vector

def freq_poi_train(target_poi, f_node, vec_len, learn_rate):
    
    poi_huffman = f_node.poi_dict[target_poi]['Huffman']
    node_vector = f_node.poi_dict[target_poi]['vector']
    e = goalong_pois_Huffman(poi_huffman, node_vector, f_node.huffman.root, vec_len, learn_rate)
    temp = e[0]
    temp = list(temp)
    temp = np.array(temp)
    node_vector += temp
    node_vector = preprocessing.normalize(node_vector)
    
    return node_vector

def category_user_train(target_poi, node_vector, category_net, category_nodes, geo_dict, hier_cate, vec_len, lamda, learn_rate, run_times):
    
    target_category = geo_dict[target_poi][0]
    
    cate_path = []
    search = False
    if target_category not in hier_cate:
        for i in hier_cate:
            if target_category in hier_cate[i]:
                search = True
                cate_path.append(i)
                break
            else:
                for j in hier_cate[i]:
                    if target_category in hier_cate[i][j]:
                        cate_path.append(i)
                        cate_path.append(j)
                        search = True
                        break
        
            if search:
                break
   
    cate_path.append(target_category)

    cate_trpath = []
    for i in range(0,len(cate_path)-1):
        j=i+1
        cate_trpath.append([cate_path[i],cate_path[j]])
      
    for i in cate_trpath:
        src = i[0]
        dest = i[1]
        e = user_category_goalong_Graph(src,dest,category_nodes,node_vector,vec_len,lamda,learn_rate,run_times)
        temp = e[0]
        temp = list(temp)
        temp = np.array(temp)
        node_vector += temp
        node_vector = preprocessing.normalize(node_vector)
        
    poi_huffman = category_nodes[target_category].poi_dict[target_poi]['Huffman']
    e = goalong_user_Huffman(poi_huffman,node_vector,category_nodes[target_category].huffman.root,vec_len,learn_rate)
    temp = e[0]
    temp = list(temp)
    temp = np.array(temp)
    node_vector += temp
    node_vector = preprocessing.normalize(node_vector)
    
    return node_vector
        
        
"""    
def train_category_vector(node_vector, target_category, category_nodes, hier_cate, vec_len, lamda, learn_rate, run_times):
    
    cate_path = []
    search = False
    if target_category not in hier_cate:
        for i in hier_cate:
            if target_category in hier_cate[i]:
                search = True
                cate_path.append(i)
                break
            else:
                for j in hier_cate[i]:
                    if target_category in hier_cate[i][j]:
                        cate_path.append(i)
                        cate_path.append(j)
                        search = True
                        break
        
            if search:
                break
   
    cate_path.append(target_category)

    cate_trpath = []
    for i in range(0,len(cate_path)-1):
        j=i+1
        cate_trpath.append([cate_path[i],cate_path[j]])
    
 
    for i in cate_trpath:
        src = i[0]
        dest = i[1]
        e = category_goalong_Graph(src,dest,category_nodes,node_vector,vec_len,lamda,learn_rate,run_times)
        temp = e[0]
        temp = list(temp)
        temp = np.array(temp)
        node_vector += temp
        node_vector = preprocessing.normalize(node_vector)
"""
class GraphWord2Vec():
    
    def __init__(self, poi_freq, graph_nodes, vec_len, learn_rate):
        
        self.vec_len = vec_len
        self.learn_rate = learn_rate
        self.the_poi_dict = None  
        self.huffman = None  
        self.gnerate_poi_dict(poi_freq, graph_nodes) 
    
    def gnerate_poi_dict(self, poi_freq, graph_nodes):

        poi_dict = {}
        if isinstance(poi_freq,dict):
            sum_count = sum(poi_freq.values())
            for poi in poi_freq:
                temp_dict = dict(poi = poi, freq = poi_freq[poi], possibility = 0 if poi_freq[poi] == 0 else poi_freq[poi]/float(sum_count), vector = graph_nodes[poi].node_vector, Huffman = None)
                poi_dict[poi] = temp_dict
        self.the_poi_dict = poi_dict

    def train_vector(self, graph_nodes, target_poi, train_set, vec_len, geo_dict, category_nodes, hier_cate, lamda, learn_rate, run_times):  

        if self.huffman==None:
            self.huffman = PoisHuffmanTree(self.the_poi_dict, vec_len=self.vec_len)  
            
        self.Deal_Gram_CBOW(graph_nodes, target_poi, train_set, vec_len, geo_dict, category_nodes, hier_cate, lamda, learn_rate, run_times)
        
        
        
    def Deal_Gram_CBOW(self, graph_nodes, target_poi, train_set, vec_len, geo_dict, category_nodes, hier_cate, lamda, learn_rate, run_times):
        
        poi_huffman = self.the_poi_dict[target_poi]['Huffman']
        
        gram_vector_sum = np.zeros([1,vec_len])
        for p in train_set:
            gram_vector_sum = gram_vector_sum+ graph_nodes[p].node_vector

        e = self.GoAlong_Huffman(poi_huffman, gram_vector_sum, self.huffman.root)
        
        
        for p in train_set:
            if p in self.the_poi_dict:
                self.the_poi_dict[p]['vector'] += e
                self.the_poi_dict[p]['vector'] = preprocessing.normalize(self.the_poi_dict[p]['vector'])
            graph_nodes[p].node_vector += e
            graph_nodes[p].node_vector = preprocessing.normalize(graph_nodes[p].node_vector)     
            train_category_vector(graph_nodes[p].node_vector,geo_dict[p][0],category_nodes,hier_cate,vec_len,lamda,learn_rate,run_times)
 
    def GoAlong_Huffman(self, poi_huffman, input_vector, root):

        node = root
        e = np.zeros([1,self.vec_len])
        for level in range(poi_huffman.__len__()):
            huffman_charat = poi_huffman[level]
            q = self.__Sigmoid(input_vector.dot(node.value.T))
            grad = self.learn_rate * (1-int(huffman_charat)-q)
            e += grad * node.value
            node.value += grad * input_vector
            node.value = preprocessing.normalize(node.value)
            if huffman_charat=='0':
                node = node.left
            else:
                node = node.right
        return e

    def __Sigmoid(self,value):
        return 1/(1+math.exp(-value))
    






