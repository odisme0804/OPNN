from Models.Tools import *
from Models.myMethods import *




if __name__=='__main__':
    
    
    fp, ds = './PoiProcessed/', 'Gowalla'
    
    vec_len = 200 # pois and edges latent factor size
    learn_rate = 0.005  # gradient decent learn rate 
    run_times = 10 # softmax iteration
    lamda = 0.01# regular term parameter
    distance_threshold = 9.75 # distance constraint of grid size
    d = 0.85 # pagerank parameter
    times = 1 # pagerank iteration
    maxiterm = 1 # gcd iter
    win_len = 5
    dist_threshold = 10
    cord_pre = 5
    ts = 6
    N=10
    
    tr_list, tu_list, te_list = dataset_splitting(fp, ds, 0.7, 0.1)
    
    
    datalist = tr_list + tu_list + te_list
    
    
    gen_pois_info(datalist, fp, ds)
    geo_dict = poi_loader(fp + ds + '_PoiInfo.txt')
    
    gen_geohash(fp, ds, cord_pre)
    gh_dict = geohash_loader(fp, ds, cord_pre)
    gh_poi = {}
    for p in gh_dict.keys():
        if gh_dict[p] not in gh_poi:
            gh_poi[gh_dict[p]] = []
        gh_poi[gh_dict[p]].append(p)
        
    #test_query = testing_data_loading_o(fp, ds, ts)

    friends_dict = gen_friends(fp,ds,datalist)

    vis = visited_dict(tr_list)
    time_nosuccessive, time_successive = gen_time_successive(tr_list,ts)
    testing_time_nosuccessive, testing_time_successive = gen_time_successive(te_list,ts)
    
    time_single_behavior, time_behavior = gen_behavior(tr_list,ts)
    
    time_user_pairs_training = {}
    gh_paiirs_training = {}

    for u in time_behavior:
        for q in time_behavior[u]:
            if len(q)>=2:
                for i in range(0,len(q)-1):
                    j = i+1
                    (p1,d1,t1) = q[i]
                    (p2,d2,t2) = q[j]
                    pre = (gh_dict[p1],d1,t1)
                    nex = (gh_dict[p2],d2,t2)
                    if p1 != p2:
                        if u not in time_user_pairs_training:
                            time_user_pairs_training[u] = []
                            
                        if u not in gh_paiirs_training:
                            gh_paiirs_training[u] = []
                        time_user_pairs_training[u].append([q[i],q[j]])
                        gh_paiirs_training[u].append([pre,nex])

    

    user_checkins = listtodict(tr_list, 0)
    a, xmin = learn_power_law(user_checkins, geo_dict)
    print(a)
    print(xmin)
    
    
    pois_freq_dict = gen_pois_frequency(tr_list, geo_dict.keys())
    b, ymin = freq_learn_power_law(user_checkins, geo_dict, pois_freq_dict)
    
    print(b)
    print(ymin)
    

    all_pois_freq_dict = gen_pois_frequency(datalist, geo_dict.keys())
    
    poi_net = poi_layer(tr_list, ts, geo_dict.keys())
    
    allpoi_net = poi_layer(datalist, ts, geo_dict.keys())
    
    grid_net = grid_layer(gh_dict, datalist, ts, gh_poi.keys())
    
    poi_check_user, user_check_poi = gen_checked_detail(tr_list)

    user_checked_freq = user_checked(tr_list)
    alluser_checked_freq = user_checked(datalist)
    print('#users : ', len(alluser_checked_freq))
    print('#pois : ', len(geo_dict))
    lmbda_dict = learn_poisson(alluser_checked_freq)
    testing_query, user_checked_dict = gen_poi_testing(datalist, te_list, ts)
    
    cal_cbow_nochange_tune(gh_paiirs_training, grid_net, gh_poi, gh_dict, poi_check_user, user_check_poi, time_nosuccessive, time_user_pairs_training, friends_dict, pois_freq_dict,  user_checked_freq, allpoi_net, alluser_checked_freq, all_pois_freq_dict, geo_dict, fp, ds, vec_len, learn_rate, lamda, run_times, maxiterm, ts, a, xmin, b, ymin, N, dist_threshold)
 
    
    