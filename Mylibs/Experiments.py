
#from Main import *
from scipy import spatial
from scipy.stats import kstest
from scipy.stats import shapiro
import numpy as np

def weekly_similarity(c1, c2):
    child = len(set(c1) & set(c2))
    mom = sqrt(len(c1)) * sqrt(len(c2))
    if mom == 0:
        return 0
    else:
        return float(child) / mom
    
def TimeSim(src_path,dataset):
    
    checkinslist = checkinstolist(src_path + dataset + '_Checkins.txt')
    user_checkins = listtodict(checkinslist, 0)
    
    weekly_list = []
    pair_count = 0
    x, y = [], []
    for u in user_checkins.keys():
        week_list = []
        weekend_list = []
        for i in range(len(user_checkins[u])):
            spec_time = None
            p1, d1, t1 = user_checkins[u][i]
            split_day = d1.split('-')
            anyday=datetime.datetime(int(split_day[0]),int(split_day[1]),int(split_day[2])).strftime("%w")
            if (int(anyday) == 0 or int(anyday) == 6):
                spec_time = 'weekend'
            else:
                spec_time = 'week'
            
            if spec_time == 'weekend':
                weekend_list.append(p1)
            else:
                week_list.append(p1)
            
        sim = weekly_similarity(week_list,weekend_list)    
        weekly_list.append(sim)
        pair_count += 1
        
    weekly_list = sorted(weekly_list)
    
    count = 0
    in_count = 0
    x, y = [], []
    for dist in weekly_list:
        count += 1
        in_count += 1
        if in_count >= 1000:
            x.append(dist)
            y.append(count / float(pair_count))
            in_count = 0
    return x, y

def weekly_graph(src_path):
    gx, gy = TimeSim(src_path, 'Gowalla')
    plt.plot(gx, gy, 'ro-', label='Gowalla', linewidth=2)
    plt.grid(True)
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=22)
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=22)
    plt.xlabel('Cosine Similarity', fontsize=28)
    plt.ylabel('Cumulative Probability', fontsize=28)
    plt.legend(fontsize=26, loc='lower right')
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.tight_layout()
    plt.show()

    
def dist_consecutive_checkins(src_path, dataset, dt=5000):
    geo_dict = poi_loader(src_path + dataset + '_PoiInfo.txt')
    checkinslist = checkinstolist(src_path + dataset + '_Checkins.txt')
    user_checkins = listtodict(checkinslist, 0)
    dist_list = []
    pair_count = 0
    for u in user_checkins.keys():
        for i in range(len(user_checkins[u]) - 1):
            p1, d1, t1 = user_checkins[u][i]
            p2, d2, t2 = user_checkins[u][i + 1]
            cate1, lng1, lat1 = geo_dict[p1]
            cate2, lng2, lat2 = geo_dict[p2]
            dist = geo_distance(lng1, lat1, lng2, lat2)
            if p1 != p2:
                if dt > dist > 0:
                    dist_list.append(dist)
                    pair_count += 1
    dist_list = sorted(dist_list)
    count = 0
    in_count = 0
    x, y = [], []
    for dist in dist_list:
        count += 1
        in_count += 1
        if in_count >= 1000:
            x.append(dist)
            y.append(count / float(pair_count))
            in_count = 0
    return x, y

def dist_graph(src_path):
    gx, gy = dist_consecutive_checkins(src_path, 'Gowalla')
    plt.plot(gx, gy, 'ro-', label='Gowalla', linewidth=2)
    plt.grid(True)
    plt.xticks(range(0, 101, 5), fontsize=22)
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=22)
    plt.xlabel('Distance (km)', fontsize=28)
    plt.ylabel('Cumulative Probability', fontsize=28)
    plt.legend(fontsize=26, loc='lower right')
    plt.axis([0, 100, 0.0, 1.0])
    plt.tight_layout()
    plt.show()



def timediff_successive_checkins(src_path, dataset):
    checkinslist = checkinstolist(src_path + dataset + '_Checkins.txt')
    user_checkins = listtodict(checkinslist, 0)
    time_list = []
    total_count = 0
    for u in user_checkins.keys():
        for i in range(len(user_checkins[u]) - 1):
            p1, d1, t1 = user_checkins[u][i]
            p2, d2, t2 = user_checkins[u][i + 1]
            td = timediff(d1, t1, d2, t2) / float(3600)
            time_list.append(td)
            total_count += 1
    time_list = sorted(time_list)
    count = 0
    in_count = 0
    x, y = [], []
    for td in time_list:
        count += 1
        in_count += 1
        if in_count >= 1000:
            x.append(td)
            y.append(count / float(total_count))
            in_count = 0
    return x, y

def time_graph(src_path):
    gx, gy = timediff_successive_checkins(src_path, 'Gowalla')
    plt.plot(gx, gy, 'ro-', label='Gowalla', linewidth=2)
    #bx, by = timediff_successive_checkins(src_path, 'Brightkite')
    #plt.plot(bx, by, 'bs-', label='Brightkite', linewidth=2)
    plt.grid(True)
    #plt.title('Time Difference between Consecutive Check-in POIs', fontsize=22)
    plt.xticks([0, 1, 2, 3, 6] + range(12, 121, 12), fontsize=22)
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=22)
    plt.xlabel('Time Difference (hours)', fontsize=28)
    plt.ylabel('Cumulative Probability', fontsize=28)
    plt.legend(fontsize=26, loc='lower right')
    plt.axis([0, 100, 0.0, 1.0])
    plt.tight_layout()
    plt.show()

def timediff_dist_checkins(src_path, dataset, dt=5000):
    geo_dict = poi_loader(src_path + dataset + '_PoiInfo.txt')
    checkinslist = checkinstolist(src_path + dataset + '_Checkins.txt')
    user_checkins = listtodict(checkinslist, 0)
    
    dist_list = []
    time_list = []
    for u in user_checkins.keys():
        for i in range(len(user_checkins[u]) - 1):
            p1, d1, t1 = user_checkins[u][i]
            p2, d2, t2 = user_checkins[u][i + 1]
            cate1, lng1, lat1 = geo_dict[p1]
            cate2, lng2, lat2 = geo_dict[p2]
            
            dist = geo_distance(lng1, lat1, lng2, lat2)
            td = timediff(d1, t1, d2, t2) / float(3600)
            if p1 != p2 and dist <=10 and int(td)<=6:
                if dt > dist > 0:
                    dist_list.append(dist)
                    time_list.append(td)
        
    return dist_list, time_list


def time_dist_graph(src_path):
    
    gx, gy = timediff_dist_checkins(src_path, 'Gowalla')
    
    plt.plot(gx, gy, 'ro', label='Gowalla', linewidth=2)
    plt.grid(True)
    plt.xticks(range(0, 11, 1), fontsize=22)
    plt.yticks(range(0, 7, 1), fontsize=22)
    plt.xlabel('Distance (km)', fontsize=28)
    plt.ylabel('Time Difference (hours)', fontsize=28)
    plt.legend(fontsize=26, loc='lower right')
    plt.axis([0, 10, 0, 6])
    plt.tight_layout()
    plt.show()
    
    
def freq_successive_checkins(src_path, dataset, time_threshold, dt=5000):
    geo_dict = poi_loader(src_path + dataset + '_PoiInfo.txt')
    checkinslist = checkinstolist(src_path + dataset + '_Checkins.txt')
    user_checkins = listtodict(checkinslist, 0)
    pois_freq_dict = gen_pois_frequency(checkinslist, geo_dict.keys())
    total_count = 0
    freq_list = []
    
    for u in user_checkins.keys():
        for i in range(len(user_checkins[u]) - 1):
            p1, d1, t1 = user_checkins[u][i]
            p2, d2, t2 = user_checkins[u][i + 1]
            if timediff(d1, t1, d2, t2) / float(3600) <= time_threshold:
                if p1 != p2:
                    cate1, lng1, lat1 = geo_dict[p1]
                    cate2, lng2, lat2 = geo_dict[p2]
                    dist = geo_distance(lng1, lat1, lng2, lat2)
                    if dt > dist > 0:
                        freq_list.append(pois_freq_dict[p2])
                        total_count += 1
    freq_list = sorted(freq_list)
    
    count = 0
    in_count = 0
    x, y = [], []
    for freq in freq_list:
        count += 1
        in_count += 1
        if in_count >= 1000:
            x.append(freq)
            y.append(count / float(total_count))
            in_count = 0
    
    return x, y
    

def freq_successive_graph(src_path,ds,ts):
    gx, gy = freq_successive_checkins(src_path, 'Gowalla', 6)
    plt.plot(gx, gy, 'ro-', label='Gowalla', linewidth=2)
    plt.grid(True)
    plt.title('check-in frequency distribution', fontsize=20)
    plt.xticks(range(0, 1000, 50), fontsize=16)
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=16)
    plt.xlabel('frequency ', fontsize=18)
    plt.ylabel('Cumulative Probability', fontsize=20)
    plt.legend(fontsize=18, loc='lower right')
    plt.axis([0, 1000, 0.0, 1.0])
    plt.show()


def dist_successive_checkins(src_path, dataset, time_threshold, dt=5000):
    
    geo_dict = poi_loader(src_path + dataset + '_PoiInfo.txt')
    checkinslist = checkinstolist(src_path + dataset + '_Checkins.txt')
    user_checkins = listtodict(checkinslist, 0)
    total_count = 0
    dist_list = []
    x, y = [], []
    for u in user_checkins.keys():
        for i in range(len(user_checkins[u]) - 1):

            p1, d1, t1 = user_checkins[u][i]
            p2, d2, t2 = user_checkins[u][i + 1]
            if timediff(d1, t1, d2, t2) / float(3600) <= time_threshold:
                if p1 != p2:
                    cate1, lng1, lat1 = geo_dict[p1]
                    cate2, lng2, lat2 = geo_dict[p2]
                    dist = geo_distance(lng1, lat1, lng2, lat2)
                    if dt > dist > 0:
                        dist_list.append(dist)
                        total_count += 1
    dist_list = sorted(dist_list)
    count = 0
    in_count = 0
    x, y = [], []
    for dist in dist_list:
        count += 1
        in_count += 1
        if in_count >= 100:
            x.append(dist)
            y.append(count / float(total_count))
            in_count = 0
    return x, y


def dist_successive_graph(src_path,ds,ts):
    gx, gy = dist_successive_checkins(src_path, 'Gowalla', ts)
    plt.plot(gx, gy, 'ro-', label='Gowalla', linewidth=2)
    plt.grid(True)
    #plt.title('Distance between Consecutive Check-in POIs in 6 hours', fontsize=20)
    plt.xticks(range(0, 101, 5), fontsize=22)
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=22)
    plt.xlabel('Distance (km)', fontsize=28)
    plt.ylabel('Cumulative Probability', fontsize=28)
    plt.legend(fontsize=26, loc='lower right')
    plt.axis([0, 100, 0.0, 1.0])
    plt.tight_layout()
    plt.show()
    
def repeat_timediff(datalist,time_threshold, fp, ds):
    
    
    geo_dict = poi_loader(fp + ds + '_PoiInfo.txt')
    
    temp_dict = {}
    user_successive = gen_successive_queryoftime(datalist, time_threshold)
    max_v = 0
    max_u = ''
    for u in user_successive:
        if max_v<len(user_successive[u]):
            max_u = u
            max_v = len(user_successive[u])
 

    for u in user_successive:
        if u == max_u:
            if u not in temp_dict:
                temp_dict[u] = {}
            for q in user_successive[u]:
                for p in range(0,len(q)):
                    if q[p][0] not in temp_dict[u]:
                        temp_dict[u][q[p][0]] = []
                    if p != 0:
                        temp_dict[u][q[p][0]].append((q[p][1],q[p][2]))
                    else:
                        temp_dict[u][q[p][0]].append((q[p][1],q[p][2]))


    time_list = []
    dis_list = []

    for u in temp_dict:
        if u == max_u:
            for p in temp_dict[u]:
                if len(temp_dict[u][p])>1:

                    for i in range(0,len(temp_dict[u][p])-1):
                        (d1,t1) = temp_dict[u][p][i]
                        j = i+1
                        (d2,t2) = temp_dict[u][p][j]
                        t_diff = abs(timediff(d1, t1, d2, t2))/float(3600)
                        time_list.append(int(t_diff))
                        for q in user_successive[u]:
                            for c in range(1,len(q)):
                                if (p,d1,t1) == q[c]:
                                    dis_list.append(int(geo_distance(geo_dict[q[c][0]][1],geo_dict[q[c][0]][2],geo_dict[q[c-1][0]][1],geo_dict[q[c-1][0]][2])))
                                
                                if (p,d2,t2) == q[c]:
                                    dis_list.append(int(geo_distance(geo_dict[q[c][0]][1],geo_dict[q[c][0]][2],geo_dict[q[c-1][0]][1],geo_dict[q[c-1][0]][2])))

                    #dis_list.append(int(dis_1))
                    #dis_list.append(int(dis_2))
    

    t_count = Counter(time_list)

    keys = t_count.keys()
    keys = sorted(keys)
    for i in range(0,72):
        if i not in t_count:
            t_count[i] = 0
    x = []
    y = []
    for key in keys:
        if key<=72:
            x.append(key)
            y.append(t_count[key])
        
    #plt.plot(x, y,'ko-',label='dis_distribution')
    bar_width = 0.4  
    plt.bar(x, y, bar_width)
    plt.savefig("time.jpg")   

    dis_count = Counter(dis_list)
    x1 = []
    y1 = []
    
    keys = dis_count.keys()
    keys = sorted(keys)
    for key in keys:
        if key != -1 and key<=1000:
            x1.append(key)
            y1.append(dis_count[key])

    plt.figure(figsize=(9,6))
    plt.scatter(x1,y1,s=25,alpha=0.4,marker='o',c='b')
    plt.savefig("dist.jpg")
    #plt.show()  


def observe_outin(datalist,poi_net,graph_nodes,fp,ds):
    
    in_dis = []
    out_dis = []
    no_dis = []
    
    user_beh = {}
    for u in datalist:
        in_co = 0
        out_co = 0
        no_edge = 0
        if u not in user_beh:
            user_beh[u] = []
            
            
        for q in datalist[u]:
            geo_dict = poi_loader(fp + ds + '_PoiInfo.txt')
            dis = geo_distance(geo_dict[q[0]][1],geo_dict[q[0]][2],geo_dict[q[1]][1],geo_dict[q[1]][2])
            if graph_nodes[q[0]].edge_vector is not None:
                if q[1] in graph_nodes[q[0]].edge_vector.keys():
                    in_co = in_co+1
                    in_dis.append(int(dis))
                else:
                    out_co= out_co+1
                    out_dis.append(int(dis))
            else:
                no_edge = no_edge+1
                no_dis.append(int(dis))
        user_beh[u] = [in_co,out_co,no_edge]
    

    
    """
    in_set = []
    out_set = []
    no_set = []
    
    for u in user_beh:
        a,b,c = user_beh[u]
        if a+b != 0:
            in_set.append(round(a/float(a+b),1))
        else:
            in_set.append(float(0))
    """
    
    

    in_counts = Counter(in_dis)
    out_counts = Counter(out_dis)

    
    
    total = 0
    for i in in_counts:
        total = total+in_counts[i]
        
    out_total = 0
    for i in out_counts:
        out_total = out_total+out_counts[i]
        
    x = []
    y = []
    z = []
    
    acur = 0
    keys = in_counts.keys()
    keys = sorted(keys)
    for key in keys:
        x.append(key)
        y.append(in_counts[key]/float(total))
        acur = acur+in_counts[key]/float(total)
        z.append(acur)   
        
    x1 = []
    y1 = []
    z1 = []
    
    acur1 = 0
    keys = out_counts.keys()
    keys = sorted(keys)
    for key in keys:
        x1.append(key)
        y1.append(out_counts[key]/float(out_total))
        acur1 = acur1+out_counts[key]/float(out_total)
        z1.append(acur1) 
    
    #bar_width = 0.4  
    #plt.bar(x, z,bar_width)
    #plt.show()
    plt.figure(figsize=(9,6))
    plt.scatter(x,z,s=25,alpha=0.4,marker='o',c='b')
    plt.scatter(x1,z1,s=25,alpha=0.4,marker='o',c='r')
    plt.show()
    
    """
    dis_counts = Counter(in_dis)
    keys = dis_counts.keys()
    keys = sorted(keys)
    x = []
    y = []
    for key in keys:
        x.append(key)
        y.append(dis_counts[key])
    
    plt.plot(x, y,'ko-',label='dis_distribution')
    plt.show()  
    """

    #print(in_dis)
    #print(out_dis)
    #print(no_dis)
    """
    a_count = Counter(in_set)
    
    x = []
    y = []
    keys = a_count.keys()
    keys = sorted(keys)
    for key in keys:
        x.append(int(key))
        y.append(int(a_count[key]))
        
    plt.figure(figsize=(9,6))
    plt.scatter(x,y,s=25,alpha=0.4,marker='o')
    plt.show()
    """

def category_poi(filepath, dataset,time_threshold):
    usercheckindict = userchecked_categorydict(filepath + dataset + '_Checkins.txt')
    for i in usercheckindict:
        pois = []
        category = []
        count_pois = 0
        count_category = 0
        for j in usercheckindict[i]:
            if usercheckindict[i][j][0] not in pois:
                count_pois = count_pois+1
                pois.append(usercheckindict[i][j][0])
                
            if usercheckindict[i][j][3] not in category:
                count_category = count_category+1
                category.append(usercheckindict[i][j][3])
                
        print(usercheckindict[i])
        print(count_pois,count_category)
        break


def user_visit(filepath, dataset,time_threshold):
    usercheckindict = usercheckeddict(filepath + dataset + '_Checkins.txt')
    user_poi = defaultdict(lambda: defaultdict(list))
    
    for u in usercheckindict:
        user_poi[u] = []
        for i in range(0,len(usercheckindict[u])):
            user_poi[u].append(usercheckindict[u][i][0])
    
    max = 0
    max_user = ""
    
    for u in usercheckindict:
        
        if len(usercheckindict[u]) > max:
            max = len(usercheckindict[u])
            max_user = u
            
    poi = {}
    for i in usercheckindict[max_user]:
        if i[0] not in poi:
            poi[i[0]] = 0
        poi[i[0]] = poi[i[0]]+1
        
    y = []
    for p in poi:
        y.append(poi[p])
        
    poi_count = Counter(y)
    
    y1 = []
    x1 = []
    keys = poi_count.keys() 
    keys = sorted(keys)
    for key in keys:
        if key<=10:
            y1.append(poi_count[key])
            x1.append(key)
 
    
    sim = []
    for u in user_poi:
        num = []
        user_count = Counter(user_poi[u])
        for i in user_count:
            num.append(user_count[i])
        num_count = Counter(num)    
       
        for c in range(1,11):
            if c not in num_count:
                num_count[c] = 0
        
        y2 = []
        keys = num_count.keys() 
        keys = sorted(keys)
        for key in keys:
            if key<=10:
                y2.append(num_count[key])

 
        result = 1 - spatial.distance.cosine(y1, y2)
        sim.append(round(result,1))

    
    sim_count = Counter(sim)
    print(sim_count)
    
    
    fig = plt.figure()  
    plt.bar(x1,y1,0.4,color="blue")  
    plt.xlabel("Visiting Frequency")  
    plt.ylabel("Count")  
    #plt.title("bar chart")  
    plt.show()    
    plt.savefig("barChart.jpg")  
    
    
def visited_transition_timediff(filepath, dataset,time_threshold):
    usercheckindict = usercheckeddict(filepath + dataset + '_Checkins.txt')
    beh = {}
    user_timediff = defaultdict(lambda: defaultdict(list))
    
    
    for u in usercheckindict:
        record = []
        for s in usercheckindict[u]:
            record.append(s)
        for i in range(0,len(usercheckindict[u])-1):
            j = i+1
            visited = []
            if usercheckindict[u][i][0] != usercheckindict[u][j][0]:
                for k in record:
                    if k[0] == usercheckindict[u][i][0] and k[1] == usercheckindict[u][i][1] and k[2] == usercheckindict[u][i][2]:
                        break
                    else:
                        visited.append(k[0]) 
                
                if usercheckindict[u][j][0] in visited:
                    
                    value = abs(timediff(usercheckindict[u][i][1],usercheckindict[u][i][2],usercheckindict[u][j][1],usercheckindict[u][j][2]))/ float(3600)
                    if value <= 50:
                        if usercheckindict[u][j][0] not in user_timediff:
                            user_timediff[usercheckindict[u][j][0]] = []    
                        user_timediff[usercheckindict[u][j][0]].append(int(value))
                        
                        
                        if usercheckindict[u][j][0] not in beh:
                            beh[usercheckindict[u][j][0]] = 1
                        else:
                            beh[usercheckindict[u][j][0]] = beh[usercheckindict[u][j][0]]+1
    max = 0
    poi = ""
    for key in beh:
        if beh[key]>max:
            max = beh[key]
            poi = key
            
    repeat_count = Counter(user_timediff[poi])
    for c in range(0,51):
        if c not in repeat_count:
            repeat_count[c] = 0
    y1 = []
    x1 = []
    keys = repeat_count.keys() 
    keys = sorted(keys)
    for key in keys:
        y1.append(repeat_count[key])
        x1.append(key)
    
    #test_stat = kstest(y1, 'norm')
    #print test_stat.pvalue
    
    sim = []
    for i in user_timediff:
        if i!=poi:
            repeat_count = Counter(user_timediff[i])
            for c in range(0,51):
                if c not in repeat_count:
                    repeat_count[c] = 0
            y2 = []
            x2 = []
            keys = repeat_count.keys() 
            keys = sorted(keys)
            for key in keys:
                y2.append(repeat_count[key])
                x2.append(key)
            result = 1 - spatial.distance.cosine(y1, y2)
            sim.append(round(result,1))
    sim_count = Counter(sim)
    print(sim_count)
    
    
    fig = plt.figure()  
    plt.bar(x1,y1,0.4,color="blue")  
    plt.xlabel("Time interval (hour)")  
    plt.ylabel("Frequency")  
    #plt.title("bar chart")  
    plt.show()    
    plt.savefig("barChart.jpg")  
    
    
def transition_timediff(filepath, dataset,time_threshold):
    usercheckindict = usercheckeddict(filepath + dataset + '_Checkins.txt')
    beh = {}
    user_timediff = defaultdict(lambda: defaultdict(list))
    
    
    for u in usercheckindict:
        record = []
        for s in usercheckindict[u]:
            record.append(s)
        for i in range(0,len(usercheckindict[u])-1):
            j = i+1
            visited = []
            if usercheckindict[u][i][0] != usercheckindict[u][j][0]:
                for k in record:
                    if k[0] == usercheckindict[u][i][0] and k[1] == usercheckindict[u][i][1] and k[2] == usercheckindict[u][i][2]:
                        break
                    else:
                        visited.append(k[0]) 
                
                if usercheckindict[u][j][0] not in visited:
                    
                    value = abs(timediff(usercheckindict[u][i][1],usercheckindict[u][i][2],usercheckindict[u][j][1],usercheckindict[u][j][2]))/ float(3600)
                    if value <= 50:
                        if usercheckindict[u][j][0] not in user_timediff:
                            user_timediff[usercheckindict[u][j][0]] = []    
                        user_timediff[usercheckindict[u][j][0]].append(int(value))
                        
                        
                        if usercheckindict[u][j][0] not in beh:
                            beh[usercheckindict[u][j][0]] = 1
                        else:
                            beh[usercheckindict[u][j][0]] = beh[usercheckindict[u][j][0]]+1
    max = 0
    poi = ""
    for key in beh:
        if beh[key]>max:
            max = beh[key]
            poi = key
            
    repeat_count = Counter(user_timediff[poi])
    for c in range(0,51):
        if c not in repeat_count:
            repeat_count[c] = 0
    y1 = []
    x1 = []
    keys = repeat_count.keys() 
    keys = sorted(keys)
    for key in keys:
        y1.append(repeat_count[key])
        x1.append(key)
    
    #test_stat = kstest(y1, 'norm')
    #print test_stat.pvalue
    
    sim = []
    for i in user_timediff:
        if i!=poi:
            repeat_count = Counter(user_timediff[i])
            for c in range(0,51):
                if c not in repeat_count:
                    repeat_count[c] = 0
            y2 = []
            x2 = []
            keys = repeat_count.keys() 
            keys = sorted(keys)
            for key in keys:
                y2.append(repeat_count[key])
                x2.append(key)
            result = 1 - spatial.distance.cosine(y1, y2)
            sim.append(round(result,1))
    sim_count = Counter(sim)
    print(sim_count)
    
    
    fig = plt.figure()  
    plt.bar(x1,y1,0.4,color="blue")  
    plt.xlabel("Time interval (hour)")  
    plt.ylabel("Frequency")  
    #plt.title("bar chart")  
    plt.show()    
    plt.savefig("barChart.jpg")  

def repaet_timediff(filepath, dataset,time_threshold):
    
    
    usercheckindict = usercheckeddict(filepath + dataset + '_Checkins.txt')
    beh = {}
    user_timediff = defaultdict(lambda: defaultdict(list))
    
    
    for u in usercheckindict:
        record = []
        for s in usercheckindict[u]:
            record.append(s)
        for i in range(0,len(usercheckindict[u])-1):
            j = i+1
            visited = []
            if usercheckindict[u][i][0] == usercheckindict[u][j][0]:
                value = abs(timediff(usercheckindict[u][i][1],usercheckindict[u][i][2],usercheckindict[u][j][1],usercheckindict[u][j][2]))/ float(3600)
                if int(value) > 0 and int(value) <= 60:
                    if usercheckindict[u][j][0] not in user_timediff:
                        user_timediff[usercheckindict[u][j][0]] = []    
                    user_timediff[usercheckindict[u][j][0]].append(int(value))
                    
            elif usercheckindict[u][i][0] != usercheckindict[u][j][0]:
                for k in record:
                    if k[0] == usercheckindict[u][i][0] and k[1] == usercheckindict[u][i][1] and k[2] == usercheckindict[u][i][2]:
                        break
                    else:
                        visited.append(k[0]) 
                
                if usercheckindict[u][j][0] in visited:
                    for t in range(j-1,-1,-1):
                        if usercheckindict[u][t][0] == usercheckindict[u][j][0]:
                            value = abs(timediff(usercheckindict[u][t][1],usercheckindict[u][t][2],usercheckindict[u][j][1],usercheckindict[u][j][2]))/ float(3600)
                            break
                    if int(value) <= 60:
                        if usercheckindict[u][j][0] not in user_timediff:
                            user_timediff[usercheckindict[u][j][0]] = []    
                        user_timediff[usercheckindict[u][j][0]].append(int(value))
                        
                        
                        if usercheckindict[u][j][0] not in beh:
                            beh[usercheckindict[u][j][0]] = 1
                        else:
                            beh[usercheckindict[u][j][0]] = beh[usercheckindict[u][j][0]]+1
    max = 0
    poi = ""
    for key in beh:
        if beh[key]>max:
            max = beh[key]
            poi = key
    repeat_count = Counter(user_timediff[poi])
    for c in range(0,61):
        if c not in repeat_count:
            repeat_count[c] = 0
    y1 = []
    x1 = []
    keys = repeat_count.keys() 
    keys = sorted(keys)
    for key in keys:
        y1.append(repeat_count[key])
        x1.append(key)
    
    test_stat = shapiro(y1)
    print(test_stat)
    
    sim = []
    all_count = 0
    normal_count = 0
    max_val = 0
    max_poi = ''
    max_vector = []
    for i in user_timediff:
        if i!=poi:
            all_count = all_count+1
            repeat_count = Counter(user_timediff[i])
            for c in range(0,61):
                if c not in repeat_count:
                    repeat_count[c] = 0
            y2 = []
            x2 = []
            keys = repeat_count.keys() 
            keys = sorted(keys)
            for key in keys:
                y2.append(repeat_count[key])
                x2.append(key)
                
            test_stat = shapiro(y2)
            if test_stat[1]>=0.05:
                print(i,y2)
                if test_stat[1]>max:
                    max_poi = i
                    max_val = test_stat[1]
                    max_vector = y2
                normal_count = normal_count+1
                
            result = 1 - spatial.distance.cosine(y1, y2)
            sim.append(round(result,1))
    sim_count = Counter(sim)
    print(sim_count)
    print(normal_count/float(all_count))
    print(max_poi)
    print(max_val)
    print(max_vector)
    
    fig = plt.figure()  
    plt.bar(x1,y1,0.4,color="blue")  
    plt.xlabel("Time interval (hour)")  
    plt.ylabel("Frequency")  
    #plt.title("bar chart")  
    plt.show()    
    plt.savefig("barChart.jpg")  
    
def repaet_timediff2(filepath, dataset,time_threshold):

    poischeckindict = poicheckeddict(filepath + dataset + '_Checkins.txt')
    pois_timediff = defaultdict(lambda: defaultdict(list))

    for p in poischeckindict:
        for i in range(0,len(poischeckindict[p])-1):
            j = i+1
            value = abs(timediff(poischeckindict[p][i][0],poischeckindict[p][i][1],poischeckindict[p][j][0],poischeckindict[p][j][1]))/ float(3600)
            if p not in pois_timediff:
                pois_timediff[p] = []   
            if int(value) > 0:
                pois_timediff[p].append(int(value)) 
        
    max_poi = ""
    max_val = 0
    for i in pois_timediff:
        y = pois_timediff[i]
        if len(y) > 3:
            test_stat = shapiro(y)
            if test_stat[1]>=0.05:
                if test_stat[1]>max_val:
                    max_val = test_stat[1]
                    max_poi = i
    
    print(max_poi)
    
    repeat_count = Counter(pois_timediff[max_poi])
    y1 = []
    x1 = []
    keys = repeat_count.keys() 
    keys = sorted(keys)
    for key in keys:
        y1.append(repeat_count[key])
        x1.append(key)
        
    fig = plt.figure()  
    plt.bar(x1,y1,0.4,color="blue")  
    plt.xlabel("Time interval (hour)")  
    plt.ylabel("Frequency")  
    #plt.title("bar chart")  
    plt.show()    
    plt.savefig("barChart.jpg")  
def cosine_similarity(v1,v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    if math.sqrt(sumxx*sumyy) == 0:
        return 0.0
    else:
        return sumxy/math.sqrt(sumxx*sumyy)

def time_diff(filepath, dataset,time_threshold):
    usercheckindict = usercheckeddict(filepath + dataset + '_Checkins.txt')

    max_user = ''
    max = 0
    for i in usercheckindict:
        if len(usercheckindict[i])>max:
            max_user = i
            max = len(usercheckindict[i])
    
    record = []
    for s in usercheckindict[max_user]:
        record.append(s)
        
        
    visited_target = []
    unvisited_target = []
    

    for i in range(0,len(usercheckindict[max_user])-1):
        j = i+1
        visited = []
        if usercheckindict[max_user][i][0]!=usercheckindict[max_user][j][0]:
            value = abs(timediff(usercheckindict[max_user][i][1],usercheckindict[max_user][i][2],usercheckindict[max_user][j][1],usercheckindict[max_user][j][2]))/ float(3600)

            if int(value) <= 0:
                for k in record:
                    if k[0] == usercheckindict[max_user][i][0] and k[1] == usercheckindict[max_user][i][1] and k[2] == usercheckindict[max_user][i][2]:
                        break
                    else:
                        visited.append(k[0])       
                value = value*60
                if usercheckindict[max_user][j][0] in visited:
                    visited_target.append(int(value))
                elif usercheckindict[max_user][j][0] not in visited:
                    unvisited_target.append(int(value))    
    

    visited_target_count = Counter(visited_target)
    unvisited_target_count = Counter(unvisited_target)
    
    print(visited_target_count)
    print(unvisited_target_count)
    
    for c in range(0,60):
        if c not in visited_target_count:
            visited_target_count[c] = 0
            
    y1 = []
    x1 = []
    keys = visited_target_count.keys() 
    keys = sorted(keys)
    for key in keys:
        y1.append(visited_target_count[key])
        x1.append(key)
    plt.plot(x1, y1,'.-')
    plt.show()
    
    for c in range(0,60):
        if c not in unvisited_target_count:
            unvisited_target_count[c] = 0
            
    y1 = []
    x1 = []
    keys = unvisited_target_count.keys() 
    keys = sorted(keys)
    for key in keys:
        y1.append(unvisited_target_count[key])
        x1.append(key)
    plt.plot(x1, y1,'.-')
    plt.show()
    
    


def poi_map_plotting(src_path, dataset):
    geo_dict = poi_loader(src_path + dataset + '_PoiInfo.txt')
    themap = Basemap(projection='gall',
                     llcrnrlon=-180,  # lower-left corner longitude
                     llcrnrlat=-90,  # lower-left corner latitude
                     urcrnrlon=180,  # upper-right corner longitude
                     urcrnrlat=90,  # upper-right corner latitude
                     resolution='l',
                     area_thresh=100000.0
                     )
    themap.drawcoastlines()
    themap.drawcountries()
    themap.fillcontinents(color='gainsboro')
    themap.drawmapboundary(fill_color='steelblue')
    for p in geo_dict.keys():
        lat, lon = geo_dict[p]
        x, y = themap(lon, lat)
        themap.plot(x, y, 'o', color='Indigo', markersize=2)
    plt.show()
