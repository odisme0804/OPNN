from Mylibs.Tools import *
import numpy as np 
import math
from sklearn import preprocessing
from sklearn.metrics.pairwise import rbf_kernel
import scipy.stats as st
import multiprocessing
import dill
import pickle
import time
import progressbar
import os.path
import code
from ast import literal_eval

class Word2Vec():
    
    def __init__(self, target_poi, poi_freq, pois_checked_dict, vec_len, learn_rate, fp, ds):
        
        self.vec_len = vec_len
        self.learn_rate = learn_rate
        self.poi_dict = None  
        self.huffman = None
        self.the_poi_dict = None
        self.gnerate_poi_dict(target_poi,poi_freq,pois_checked_dict,fp,ds, vec_len, learn_rate) 
        
    def gnerate_poi_dict(self, target_poi, poi_freq, pois_checked_dict, fp, ds, vec_len, learn_rate):
    
        poi_dict = {}
        if isinstance(poi_freq,dict):

            sum_count = sum(poi_freq.values())

            temp_dict = dict(poi = target_poi,freq = poi_freq[target_poi],possibility = poi_freq[target_poi]/float(sum_count) if sum_count!=0 else float(0),vector = np.random.random([1,self.vec_len]),Huffman = None)
            poi_dict[target_poi] = temp_dict
        self.poi_dict = poi_dict


class TreeNode():
    
    def __init__(self, x, left,right,parent):
        self.val = x
        self.left = left
        self.right = right
        self.parent = parent


def make_tree(arr, parent):
    if not arr:
        return None

    length = len(arr)
    if length == 1:
        return TreeNode(arr[0], None, None, parent)
    else:
        mid = int(len(arr)/2)
        mid_node = TreeNode(arr[mid], None, None, parent)
        mid_node.left = make_tree(arr[0:mid], mid_node)
        mid_node.right = make_tree(arr[mid+1:length], mid_node)
        return mid_node


        
class HuffmanTreeNode():
    
    def __init__(self,value,freq):
        
        self.possibility = freq
        self.left = None
        self.right = None
        self.value = value 
        self.Huffman = ""


class HuffmanTree():
   
    def __init__(self, target_poi, loc, poi_dict, poi_freq, pe_dict, geo_dict, precision, cord_tree ,vec_len):
        self.vec_len = vec_len
        self.root = None
        self.head = None
        
        temp_pe_dict = {}
        temp_pe_dict[target_poi] = pe_dict[target_poi][:-2]+loc
       
        ptr = cord_tree
        for i in range(0,len(temp_pe_dict[target_poi])):
            if temp_pe_dict[target_poi][i] == '0':
                ptr = ptr.left
            elif temp_pe_dict[target_poi][i] == '1':
                ptr = ptr.right
        
        if ptr.left is None and ptr.right is None:
            poi_dict_list = list(poi_freq.keys())
            node_list = [HuffmanTreeNode(x,poi_freq[x]) for x in poi_dict_list]
            head,huffman_root = self.build_tree(target_poi,node_list,cord_tree,temp_pe_dict)
            self.generate_huffman_code(head, huffman_root , poi_dict, temp_pe_dict, target_poi) 
        else:
            self.head = cord_tree
            self.generate_huffman_code_exist(cord_tree , poi_dict, temp_pe_dict, target_poi) 
          
    
    def build_tree(self, target_poi, node_list, cord_tree, temp_pe_dict):
        

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

        ptr = cord_tree
        for i in range(0,len(temp_pe_dict[target_poi])):
            if temp_pe_dict[target_poi][i] == '0':
                ptr = ptr.left
            elif temp_pe_dict[target_poi][i] == '1':
                ptr = ptr.right

        ptr.left = self.root
        ptr.right = self.root
        
        self.head = cord_tree
        

        return cord_tree, self.root
        
        

    def generate_huffman_code(self, head, huffman_root, poi_dict, temp_pe_dict, target_poi):
        
        stack = [huffman_root]
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
            if poi == target_poi:
                poi_dict[poi]['Huffman'] = temp_pe_dict[poi]+"0"+code
                break

            
    def generate_huffman_code_exist(self, head, poi_dict, temp_pe_dict, target_poi):
        

        ptr = head
        for i in range(0,len(temp_pe_dict[target_poi])):

            if temp_pe_dict[target_poi][i] == '0':
                ptr = ptr.left
            elif temp_pe_dict[target_poi][i] == '1':
                ptr = ptr.right

        ptr = ptr.left
  
        paths = self.find_path(ptr, target_poi, code="")
        
        code = ""
        for i in paths:
            for j in i:
                if isinstance(j[0],str) and j[0] == target_poi:             
                    code = j[1]
                    break
            if len(code)!=0:
                break
        
        poi_dict[target_poi]['Huffman'] = temp_pe_dict[target_poi]+"0"+code
        
        
        #paths = self.find_path(head, target_poi, code="")
        """
        code = ""
        for i in paths:
            for j in i:
                if isinstance(j[0],str) and j[0] == target_poi:             
                    code = j[1]
                    break
            if len(code)!=0:
                break
        
        print(poi_dict[target_poi]['Huffman'])
        print(code)
        """
        
    def find_path(self, tree, target_poi, code):
        paths = []
        if not (tree.left or tree.right):
            return [[(tree.value,code)]]
        if tree.left:
            paths.extend([[(tree.value,'left')] + child for child in self.find_path(tree.left,target_poi,code+'0')])
        if tree.right:
            paths.extend([[(tree.value,'right')] + child for child in self.find_path(tree.right,target_poi,code+'1')])  
        
        return paths
    
    
    def merge(self,node1,node2):
        top_pos = node1.possibility + node2.possibility
        top_node = HuffmanTreeNode(np.zeros([1,self.vec_len]), top_pos)
        #top_node = HuffmanTreeNode(np.random.random([1,self.vec_len]), top_pos)

        if node1.possibility >= node2.possibility :
            top_node.left = node2
            top_node.right = node1
        else:
            top_node.left = node1
            top_node.right = node2
        return top_node


class PoisCounter():
    
    def __init__(self, data,pois_checked_dict):
        self.data = data
        self.count_res = None
        self.Pois_Count(self.data,pois_checked_dict)


    def Pois_Count(self,data,pois_checked_dict):
        

        selectpois_dict = {}
        for i in data[0]:
            selectpois_dict[i] = pois_checked_dict[i]

        self.count_res = selectpois_dict
        

def makeList(tree):
    paths = []
    if not (tree.left or tree.right):
        return [[tree.possibility]]
    if tree.left:
        paths.extend([[(tree.possibility,'left')] + child for child in makeList(tree.left)])
    if tree.right:
        paths.extend([[(tree.possibility,'right')] + child for child in makeList(tree.right)])  
    return paths



    
def add_node(root,path,node,vec_len,index):
    
    if isinstance(path[index], tuple):
        if path[index][1] == 'left':
            add_node(root.left,path,node,vec_len,index+1)
        if path[index][1] == 'right':
            add_node(root.right,path,node,vec_len,index+1)
    else:
        if root.left is None and root.right is None:
            hnode1 = HuffmanTreeNode(np.zeros([1,vec_len]), float(node))
            root.left = hnode1
            hnode2 = HuffmanTreeNode(np.zeros([1,vec_len]), float(node))
            root.right = hnode2

def binary_insert(root, node_list, vec_len, digitial_type):
    

        
    path = makeList(root)

    for j in node_list:
        for p in path:
            addin = []
            for i in range(len(p)):
                if i%2 == digitial_type and isinstance(p[i],tuple):
                    if p[i][1] == 'left':
                        if j <= p[i][0]:
                            addin.append(1)
                        else:
                            addin.append(0)
                    else:
                        if j > p[i][0]:
                            addin.append(1)
                        else:
                            addin.append(0)
            
            temp = list(set(addin))                 
            if len(list(set(addin))) == 1 and temp[0]==1:
                add_node(root,p,j,vec_len,index=0)
          
def build_CBT(precision,vec_len):
    
    lng = [float(-180),float(180)]
    lat = [float(-90),float(90)]

    repeat_count = precision/2
    
    level = 1
    while level<=repeat_count+1:
        temp_lng_list = []
        temp = (abs(lng[0])+abs(lng[1]))/float(pow(2,level))
        start_value = lng[0]
        while start_value<lng[1]-temp:
            start_value = start_value+temp
            temp_lng_list.append(start_value)
        temp_lng_list.sort()
        level = level+1
    

    level = 1
    while level<=repeat_count:
        temp_lat_list = []
        temp = (abs(lat[0])+abs(lat[1]))/float(pow(2,level))
        start_value = lat[0]
        while start_value<lat[1]-temp:
            start_value = start_value+temp
            temp_lat_list.append(start_value)
        temp_lat_list.sort()
        level = level+1

    lng_tree = make_tree(temp_lng_list,None)       
    lat_tree = make_tree(temp_lat_list,None)
    lng_list = traverse(lng_tree)
    lat_list = traverse(lat_tree)

       
    root_node = HuffmanTreeNode(np.zeros([1,vec_len]), float(0))
    leftnode = HuffmanTreeNode(np.zeros([1,vec_len]), float(0))
    rightnode = HuffmanTreeNode(np.zeros([1,vec_len]), float(0))
    head = root_node
    root_node.left = leftnode
    root_node.right = rightnode
    lng_list = lng_list[2:]
    lat_list = lat_list[2:]
    

    lng_index = 0
    lat_index = 0

    while True:
        temp_list = []
        for i in range(lng_index,len(lng_list)):
            if lng_list[i] != '#':
                temp_list.append(lng_list[i])
            else:
                binary_insert(head,temp_list,vec_len,digitial_type=0)
                temp_list = []
                lng_index = i+1
                for j in range(lat_index,len(lat_list)):
                    if lat_list[j]!= '#':
                        temp_list.append(lat_list[j])
                    else:
                        binary_insert(head,temp_list,vec_len,digitial_type=1)
                        temp_list = []
                        lat_index = j+1
                        break   
        
        if lng_index == len(lng_list) and lat_index == len(lat_list):
            break       
  
    return head


def gram_cbow(pe_dict, train_set, target_poi, info_p, head, fp, ds, precision, vec_len, learn_rate):
    

    
    gram_vector_sum = np.zeros([1,vec_len])
    for p in train_set:
        loc = pe_dict[p][-2:]
        gram_vector_sum = gram_vector_sum+info_p[p][loc].poi_dict[p]['vector']
    
    gram_vector_sum = gram_vector_sum/len(train_set)
    
    target_loc = pe_dict[target_poi][-2:]
    poi_huffman = info_p[target_poi][target_loc].poi_dict[target_poi]['Huffman']
        
    head, e = goalong_Huffman(poi_huffman,gram_vector_sum,head,vec_len,learn_rate)
        
    for p in train_set:
        loc = pe_dict[p][-2:]
        info_p[p][loc].poi_dict[p]['vector']+= e
        info_p[p][loc].poi_dict[p]['vector'] = preprocessing.normalize(info_p[p][loc].poi_dict[p]['vector'])
                
        
    return info_p, head

      
def goalong_Huffman(poi_huffman, input_vector, root, vec_len, learn_rate):
    

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
    return root, e

def user_goalong_Huffman(poi_huffman, input_vector, root, vec_len, learn_rate):
    

    node = root
    e = np.zeros([1,vec_len])
    for level in range(len(poi_huffman)):
        huffman_charat = poi_huffman[level]
        q = Sigmoid(np.dot(input_vector,node.value.T))
        grad = learn_rate * (int(huffman_charat)-q)
        e += grad * node.value
        #node.value += (grad * input_vector)
        #node.value = preprocessing.normalize(node.value)
        if huffman_charat=='0':
            node = node.left
        else:
            node = node.right
    return e

def Sigmoid(value):
    return 1/(1+math.exp(-value))


def join_vec(rootnode,vec_len):

    temp = []
    thislevel = [rootnode]
    while thislevel:
        nextlevel = list()
        for n in thislevel:
            n.value = np.zeros([1,vec_len])
            temp.append(n.value)
            if n.left: 
                nextlevel.append(n.left)
            if n.right: 
                nextlevel.append(n.right)
        temp.append('#')
        thislevel = nextlevel
    return rootnode

def single_node_cbow(pe_dict, info_p, loc, target_poi, head, vec_len, learn_rate, filepath, dataset, precision):
    
    
    target_loc = pe_dict[target_poi][-2:]
    poi_huffman = info_p[target_poi][loc].poi_dict[target_poi]['Huffman']
    gram_vector = info_p[target_poi][target_loc].poi_dict[target_poi]['vector']
    head, e = goalong_Huffman(poi_huffman, gram_vector, head, vec_len, learn_rate)
    info_p[target_poi][target_loc].poi_dict[target_poi]['vector'] += e
    info_p[target_poi][target_loc].poi_dict[target_poi]['vector'] = preprocessing.normalize(info_p[target_poi][loc].poi_dict[target_poi]['vector'])
    return info_p, head
            
def discovery_vector(pois_checked_dict, filepath, dataset, precision, vec_len, learn_rate):
    
    pe_dict = poi_encode_loader(filepath,dataset,precision)
    info_p = defaultdict(lambda: defaultdict(list))
    geo_dict = poi_loader(filepath + dataset + '_PoiInfo.txt')
    
    for p in geo_dict.keys():
        if p not in pois_checked_dict:
            pois_checked_dict[p] = 0
    
    print('cordinate_tree create....')

    tree_file = os.path.exists(filepath + dataset + '_Cordtree_'+str(precision)+'.dat')
    treepath = filepath + dataset + '_Cordtree_'+str(precision)+'.dat'
    
    if tree_file:
        with open(treepath, 'r') as inf:
            cord_tree = pickle.load(inf)
    else:
        cord_tree = build_CBT(precision,vec_len)
        with open(treepath, 'w') as outf:
            pickle.dump(cord_tree, outf)

    cord_tree = join_vec(cord_tree,vec_len)
    head = cord_tree
    print('huffman_tree create....')
    
    
    temp_region_code = {}
    for p in geo_dict.keys():
        if p not in temp_region_code:
            temp_region_code[p] = {}
        #encode_cordinate(p,head,filepath,dataset,precision)
        temp_region = gen_grid_pois(p,filepath,dataset,precision)
        for i in temp_region:
            query_encode = pe_dict[p][:-2]+i
            if i not in temp_region_code[p]:
                temp_region_code[p][i] = []
            temp_region_code[p][i] = query_encode
            
    for p in geo_dict.keys():
        query_huffman_pois = {}
        query_poi = p
        temp_region = gen_grid_pois(query_poi,filepath,dataset,precision)

        for i in temp_region:
            query_encode = pe_dict[query_poi][:-2]+i
            region_poi = []
            
            for j in temp_region_code:
                for loc in temp_region_code[j]:
                    if loc == i and temp_region_code[j][loc] == query_encode:
                        region_poi.append(j)
            
          
                        
            if query_poi not in region_poi:
                region_poi.append(query_poi)
    
            query_huffman_pois.setdefault(i,[]).append(region_poi)  
        target_poi = p
        for loc in query_huffman_pois.keys():
            pc = PoisCounter(query_huffman_pois[loc],pois_checked_dict)
            info_p[p][loc] = Word2Vec(p,pc.count_res,pois_checked_dict,vec_len,learn_rate,filepath, dataset)
            info_p[p][loc].huffman = HuffmanTree(p,loc,info_p[p][loc].poi_dict,pc.count_res,pe_dict,geo_dict,precision,cord_tree,vec_len)
            if loc == pe_dict[target_poi][-2:]:
                info_p, head = single_node_cbow(pe_dict, info_p, loc, target_poi, head, vec_len, learn_rate, filepath, dataset, precision)
    
    return info_p, head
    


def area(loc1,poi_lng,poi_lat,minlng,maxlng,minlat,maxlat,area_type,precision):    
    
    code1 = encode(minlng,minlat,precision)
    code2 = encode(maxlng,minlat,precision)
    code3 = encode(minlng,maxlat,precision)
    code4 = encode(maxlng,maxlat,precision)  
    
    code1 = code1[-2:]
    code2 = code2[-2:]
    code3 = code3[-2:]
    code4 = code4[-2:]
   
    
    
    lng = [float(-180),float(180)]    
    lat = [float(-90),float(90)]

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
    
    if area_type == 4:
        dis1 = geo_distance(minlng,minlat,mid_lng,minlat)
        dis2 = geo_distance(minlng,minlat,minlng,mid_lat)
        area1 = dis1*dis2
        
        dis3 = geo_distance(maxlng,minlat,mid_lng,minlat)
        dis4 = geo_distance(maxlng,minlat,maxlng,mid_lat)
        area2 = dis3*dis4
        
        dis5 = geo_distance(maxlng,maxlat,maxlng,mid_lat)
        dis6 = geo_distance(maxlng,maxlat,mid_lng,maxlat)
        area3 = dis5*dis6
        
        dis7 = geo_distance(minlng,maxlat,minlng,mid_lat)
        dis8 = geo_distance(minlng,maxlat,mid_lng,maxlat)
        area4 = dis7*dis8
        

        if loc1 == code1:
            return area1/float(area1+area2+area3+area4)
        elif loc1 == code2:
            return area2/float(area1+area2+area3+area4)
        elif loc1 == code3:
            return area3/float(area1+area2+area3+area4)
        elif loc1 == code4:
            return area4/float(area1+area2+area3+area4)
    else:
        if code1 == code2 and code3 == code4:
            dis1 = geo_distance(minlng,minlat,maxlng,minlat)
            dis2 = geo_distance(minlng,minlat,minlng,mid_lat)
            area1 = dis1*dis2
            
            dis3 = geo_distance(minlng,maxlat,maxlng,maxlat)
            dis4 = geo_distance(minlng,maxlat,minlng,mid_lat)
            area2 = dis3*dis4
            
            if loc1 == code1:
                return area1/float(area1+area2)
            elif loc1 == code3:
                return area2/float(area1+area2)
            
        elif code1 == code3 and code2 == code4:
            dis1 = geo_distance(minlng,minlat,mid_lng,minlat)
            dis2 = geo_distance(minlng,minlat,minlng,maxlat)
            area1 = dis1*dis2
            
            dis3 = geo_distance(maxlng,minlat,mid_lng,minlat)
            dis4 = geo_distance(maxlng,minlat,maxlng,maxlat)
            area2 = dis3*dis4
            
            if loc1 == code1:
                return area1/float(area1+area2)
            elif loc1 == code2:
                return area2/float(area1+area2)
        
def cal_area(loc1,target_poi,filepath,dataset,precision):
    geo_dict = poi_loader(filepath + dataset + '_PoiInfo.txt')
    
    region = gen_grid_pois(target_poi,filepath,dataset,precision)
    
    if len(region) == 1:
        return float(1)
    else:
        minLng,maxLng,minLat,maxLat = gen_grid_pois2(target_poi,filepath,dataset,precision)
        prod = area(loc1,geo_dict[target_poi][1],geo_dict[target_poi][2],minLng,maxLng,minLat,maxLat,len(region),precision)
        return prod

def cal_candidate_prob_context(queue, data, u_vec, recent_poi_vec, pois_vec):
    
    prob_val = {}

    for p in data:

        passion = np.dot(u_vec[0], pois_vec[p][0].T)
        recent = np.dot(recent_poi_vec[0], pois_vec[p][0].T)
        prob_val[p] = passion+recent


    queue.put(prob_val)
    
def cal_candidate_prob(queue, data, u_vec, pois_vec):
    
    prob_val = {}

    for p in data:
        passion = np.dot(u_vec[0], pois_vec[p][0].T)
        prob_val[p] = passion
        
    queue.put(prob_val)

def cal_trainpois_prob(queue, data, u_vec, pois_vec):

    prob_val = {}

    for p in data:
        passion = math.exp(np.dot(u_vec[0], pois_vec[p][0].T))  
        prob_val[p] = passion
        
    queue.put(prob_val)

def user_gram_cbow(pe_dict, u, user_vector, target_poi, info_p, head, filepath, dataset, precision, vec_len, learn_rate):


    target_loc = pe_dict[target_poi][-2:]
    poi_huffman = info_p[target_poi][target_loc].poi_dict[target_poi]['Huffman']
    gram_vector = user_vector[u]
    e = user_goalong_Huffman(poi_huffman, gram_vector, head, vec_len, learn_rate)
    user_vector[u] += e
    user_vector[u] = preprocessing.normalize(user_vector[u])
    return user_vector[u]


def cal_train_prob(pe_dict, train_set, target_poi, info_p, head, filepath, dataset, precision, vec_len, N, learn_rate, maxiterm):
    
    
    #geo_dict = poi_loader(filepath + dataset + '_PoiInfo.txt')
    
                
    prob = 1.0
    gram_vector_sum = np.zeros([1,vec_len])
    for p in train_set:    
        loc = pe_dict[p][-2:]
        gram_vector_sum = gram_vector_sum+info_p[p][loc].poi_dict[p]['vector']
    
    gram_vector_sum = gram_vector_sum/len(train_set)
    
    prob_val = {}

    c = target_poi   
    for i in info_p[c].keys():
        target_loc = i
        area = cal_area(target_loc,c,filepath,dataset,precision)
        code = info_p[c][target_loc].poi_dict[c]['Huffman']
        ptr = head
        
        sigmoid_node_value = 1.0
        for t in range(0,len(code)):
            if t!= precision:
                if code[t] == '0':
                    sigmoid_node_value = sigmoid_node_value*Sigmoid(np.dot(gram_vector_sum[0], ptr.value[0].T))
                    ptr = ptr.left
                else:
                    sigmoid_node_value = sigmoid_node_value*(1-Sigmoid(np.dot(gram_vector_sum[0], ptr.value[0].T)))  
                    ptr = ptr.right  
            else:
                ptr = ptr.left
        
        """        
        for t in range(0,precision):
            if code[t] == '0':
                ptr = ptr.left
                sigmoid_node_value = sigmoid_node_value*float((1-Sigmoid(gram_vector_sum[0].dot(ptr.value[0].T))))
            else:
                ptr = ptr.right
                sigmoid_node_value = sigmoid_node_value*float((Sigmoid(gram_vector_sum[0].dot(ptr.value[0].T))))    
                
                
        if len(code) == precision:
            pass
        else:
            for co in range(precision,len(code)-1):
                if code[co] == '0':
                    ptr = ptr.left
                else:
                    ptr = ptr.right
                        
            sigmoid_node_value = sigmoid_node_value*float((Sigmoid(gram_vector_sum[0].dot(ptr.value[0].T))))
        """   
                
        prob = prob*area*sigmoid_node_value
    prob_val[c] = prob
            

    return prob_val[c]   
        
        


def cal_testing_context(pe_dict, geo_dict, info_p, user_vector, u, filepath, dataset, precision, vec_len, N, final_poi, a, xmin):
    

    candidate_poi = geo_dict.keys()
      
    prob_val = {}
    
    loc1 = pe_dict[final_poi][-2:]
    
    for p in candidate_poi:
        
        loc2 = pe_dict[p][-2:]
        recent = np.dot(info_p[final_poi][loc1].poi_dict[final_poi]['vector'][0], info_p[p][loc2].poi_dict[p]['vector'][0].T)
        passion = np.dot(user_vector[u][0], info_p[p][loc2].poi_dict[p]['vector'][0].T)
        prob_val[p] = passion+recent
  
    prob_val = cal_normalize(prob_val)

    res = list(sorted(prob_val, key=prob_val.__getitem__, reverse=True))
    top_n = res[:N]
 
    return top_n



def cal_testing(pe_dict, geo_dict, info_p, user_vector, u, filepath, dataset, precision, vec_len, N, a, xmin):


    candidate_poi = geo_dict.keys()

    prob_val = {}

    for p in candidate_poi:
        loc = pe_dict[p][-2:]
        passion = np.dot(user_vector[u][0], info_p[p][loc].poi_dict[p]['vector'][0].T)
        prob_val[p] = passion
  
    prob_val = cal_normalize(prob_val)

    
    res = list(sorted(prob_val, key=prob_val.__getitem__, reverse=True))
    top_n = res[:N]

    return top_n


def testing_userans(all_pois_freq_dict, user_checked_freq, context, pe_dict, geo_dict, info_p, user_vector, u, filepath, dataset, precision, vec_len, ts, N, a, xmin, alpha, beta):
    
    [(q_poi, q_day, q_time)] = context
    loc1 = pe_dict[q_poi][-2:]

    
    visited_poi = user_checked_freq[u].keys()
    
    candidate_poi = list(set(dist_filter(q_poi, geo_dict, 100)) - set(visited_poi))
    #candidate_poi = list(set(geo_dict.keys())-set(visited_poi))
    user_val = {}
    recent_val = {}
    will_val = {}
    freq_val = {}
    
    for p in candidate_poi:
        
        loc2 = pe_dict[p][-2:]
        recent_val[p] = np.dot(info_p[q_poi][loc1].poi_dict[q_poi]['vector'][0], info_p[p][loc2].poi_dict[p]['vector'][0].T)
        user_val[p] = np.dot(user_vector[u][0], info_p[p][loc2].poi_dict[p]['vector'][0].T)
        
        dis = geo_distance(geo_dict[q_poi][1],geo_dict[q_poi][2],geo_dict[p][1],geo_dict[p][2])
        if dis == 0.0:
            dis = 0.01
            
        will_val[p] = willness(a, xmin, dis)
        freq_val[p] = all_pois_freq_dict[p]

    
    prob_user_val = cal_normalize(user_val)
    prob_recent_val = cal_normalize(recent_val)
    prob_will_val = cal_normalize(will_val)
    prob_freq_val = cal_normalize(freq_val)
    
    prob_val = {}
    
    
   
    for i in candidate_poi:
        #prob_val[i] = alpha*(prob_user_val[i]+prob_recent_val[i])+beta*prob_will_val[i]
        prob_val[i] = prob_user_val[i]+prob_recent_val[i]

      
    res = list(sorted(prob_val, key=prob_val.__getitem__, reverse=True))
    top_n = res[:N]
    

    return top_n


def gcd_content(pe_dict, context, wr_dict, head, target_poi, vec_len, learn_rate, filepath, dataset, precision):
    
    gram_vector_sum = np.zeros([1,vec_len])
    
    for p in context:    
        loc = pe_dict[p][-2:]
        gram_vector_sum = gram_vector_sum+wr_dict[p][loc]['vector']
    
    #gram_vector_sum = gram_vector_sum/len(context)

    
    prob = 1.0
    for i in wr_dict[target_poi].keys():
        target_loc = i
        area = cal_area(target_loc,target_poi,filepath,dataset,precision)
        code = wr_dict[target_poi][target_loc]['Huffman']
        ptr = head
        sigmoid_node_value = 1.0
        for t in range(0,len(code)):
            if t!= precision:
                if code[t] == '0':
                    sigmoid_node_value = sigmoid_node_value*Sigmoid(np.dot(gram_vector_sum[0], ptr.value[0].T))
                    ptr = ptr.left
                else:
                    sigmoid_node_value = sigmoid_node_value*(1-Sigmoid(np.dot(gram_vector_sum[0], ptr.value[0].T)))  
                    ptr = ptr.right  
            else:
                ptr = ptr.left
        prob = prob*area*sigmoid_node_value

    return wr_dict, head, prob

def gcd_user(pe_dict, info_p, head, target_poi, user_checked_freq, u, user_vector, vec_len, learn_rate, filepath, dataset, precision):

    user_vector[u] = user_gram_cbow(pe_dict, u, user_vector, target_poi, info_p, head, filepath, dataset, precision, vec_len, learn_rate)
    

    prob_val = {}
                    
    for p in user_checked_freq[u].keys():
        loc = pe_dict[p][-2:]
        prob_val[p] = math.exp(np.dot(user_vector[u][0], info_p[p][loc].poi_dict[p]['vector'][0].T))  
  
    prob_val = cal_normalize(prob_val)
    prob_passion = prob_val[target_poi]
 

    return user_vector[u], prob_passion






    
def poi2vec(all_pois_freq_dict, user_checked_freq, train_dict, testing_dict, nosuccessive_train_dict, nosuccessive_testing_dict, raw_tainlist, user_vector, alluser_checked_freq, filepath, dataset, precision, ts, N, vec_len, learn_rate, maxiterm, epsilon, a, xmin, win_len):
    
    # generate POI checkin dict: Dict[POI] = "checkin count"
    pois_checked_dict = pois_count(raw_tainlist)
	
    geo_dict = poi_loader(filepath + dataset + '_PoiInfo.txt')

    pe_dict = poi_encode_loader(filepath, dataset, precision)
    
    info_p, head = discovery_vector(pois_checked_dict,filepath,dataset,precision,vec_len,learn_rate)
        
    vi_count = 0
    for u in train_dict:
            vi_count = vi_count+len(train_dict[u])
            
           
    print("cbow.....")
    
    bar2 = progressbar.ProgressBar(widgets=[progressbar.Percentage(),progressbar.Bar()],maxval=vi_count).start()
    p_n2 = 0  
          
    for u in train_dict:
        for q in train_dict[u]:
            before, target_poi, after = q
            context = list(set(before))
            gruthset = list(set(after))
            if len(context)!=0:
                info_p, head = gram_cbow(pe_dict, context, target_poi, info_p, head, filepath, dataset, precision, vec_len, learn_rate)
                prob_content = cal_train_prob(pe_dict, context, target_poi, info_p, head, filepath, dataset, precision, vec_len, N, learn_rate, maxiterm)
                 
                for iter_num in range(maxiterm):
                    if iter_num == 0:
                        initial_prob = prob_content
                    
                    if prob_content < 1.0:
 
                        for p in context:
                            loc = pe_dict[p][-2:]
                            info_p[p][loc].poi_dict[p]['vector'] -=  learn_rate*(prob_content-1)*info_p[p][loc].poi_dict[p]['vector']
                            info_p[p][loc].poi_dict[p]['vector'] = preprocessing.normalize(info_p[p][loc].poi_dict[p]['vector'])
                                
                        info_p, head = gram_cbow(pe_dict, context, target_poi, info_p, head, filepath, dataset, precision, vec_len, learn_rate)
                        prob_content = cal_train_prob(pe_dict, context, target_poi, info_p, head, filepath, dataset, precision, vec_len, N, learn_rate, maxiterm)
                            
                            
                        if prob_content < initial_prob:
                            for p in context:
                                loc = pe_dict[p][-2:]
                                info_p[p][loc].poi_dict[p]['vector'] +=  learn_rate*(1-prob_content)*info_p[p][loc].poi_dict[p]['vector']
                                info_p[p][loc].poi_dict[p]['vector'] = preprocessing.normalize(info_p[p][loc].poi_dict[p]['vector'])
                                    
                            info_p, head = gram_cbow(pe_dict, context, target_poi, info_p, head, filepath, dataset, precision, vec_len, learn_rate)
                            break
                        else:
                            initial_prob = prob_content  
                      
            else:
                    
                user_vector[u] = user_gram_cbow(pe_dict, u, user_vector, target_poi, info_p, head, filepath, dataset, precision, vec_len, learn_rate)
                    
                prob_val = {}
                for p in alluser_checked_freq[u].keys():
                    loc = pe_dict[p][-2:]
                    prob_val[p] = math.exp(np.dot(user_vector[u][0], info_p[p][loc].poi_dict[p]['vector'][0].T))  
            
                prob_val = cal_normalize(prob_val)
                prob_passion = prob_val[target_poi]
                
       
                for iter_num in range(maxiterm):
                    if iter_num == 0:
                        initial_prob = prob_passion
                    
                    if prob_passion < 1.0:
                            
                        user_vector[u] -= learn_rate*(prob_passion-1)*user_vector[u]
                        user_vector[u] = preprocessing.normalize(user_vector[u])
                        user_vector[u], prob_passion = gcd_user(pe_dict, info_p, head, target_poi, alluser_checked_freq, u, user_vector, vec_len, learn_rate, filepath, dataset, precision)
                            
                        if prob_passion < initial_prob:
                            user_vector[u] += learn_rate*(1-prob_passion)*user_vector[u]
                            user_vector[u] = preprocessing.normalize(user_vector[u])
                            user_vector[u], prob_passion = gcd_user(pe_dict, info_p, head, target_poi, alluser_checked_freq, u, user_vector, vec_len, learn_rate, filepath, dataset, precision)
                            break
                        else:
                            initial_prob = prob_passion

                  
                    
            time.sleep(0.01)
            p_n2 = p_n2+1
            bar2.update(p_n2)

    bar2.finish()   
    
    vi_count = 0
    for u in nosuccessive_train_dict:
        vi_count = vi_count+len(nosuccessive_train_dict[u])
    
      
    bar3 = progressbar.ProgressBar(widgets=[progressbar.Percentage(),progressbar.Bar()],maxval=vi_count).start()
    p_n3 = 0  
            
    print('user vector training...')
  
    for u in nosuccessive_train_dict:
        for target_poi in nosuccessive_train_dict[u]:
            user_vector[u] = user_gram_cbow(pe_dict, u, user_vector, target_poi, info_p, head, filepath, dataset, precision, vec_len, learn_rate)
            prob_val = {}
            for p in alluser_checked_freq[u].keys():
                loc = pe_dict[p][-2:]
                prob_val[p] = math.exp(np.dot(user_vector[u][0], info_p[p][loc].poi_dict[p]['vector'][0].T))  
            
            prob_val = cal_normalize(prob_val)
            prob_passion = prob_val[target_poi]
            
        
            for iter_num in range(maxiterm):
                
                if iter_num == 0:
                    initial_prob = prob_passion
                    
                if prob_passion < 1.0: 
                    
                    user_vector[u] -= learn_rate*(prob_passion-1)*user_vector[u]
                    user_vector[u] = preprocessing.normalize(user_vector[u])
                    user_vector[u], prob_passion = gcd_user(pe_dict, info_p, head, target_poi, alluser_checked_freq, u, user_vector, vec_len, learn_rate, filepath, dataset, precision)
                            
                    if prob_passion < initial_prob:
                        user_vector[u] += learn_rate*(1-prob_passion)*user_vector[u]
                        user_vector[u] = preprocessing.normalize(user_vector[u])
                        user_vector[u], prob_passion = gcd_user(pe_dict, info_p, head, target_poi, alluser_checked_freq, u, user_vector, vec_len, learn_rate, filepath, dataset, precision)
                        break
                    else:
                        initial_prob = prob_passion
                            
                    
            time.sleep(0.01)
            p_n3 = p_n3+1
            bar3.update(p_n3)

    bar3.finish()      
    
    """
    rg_file = open(filepath + dataset + '_Users_vector' + '.txt', 'w')
    for u in user_vector.keys():   
        vec = user_vector[u]
        rg_file.write(u+'\t')
        rg_file.write("\t".join(str(elem) for elem in vec[0]))
        rg_file.write('\n')
    rg_file.close()
    
    
    candidate_poi = geo_dict.keys()
    rg_file = open(filepath + dataset + '_Pois_vector' + '.txt', 'w')
    all_vec = {}
    for y in candidate_poi:
        all_vec[y] = {}
        for i in info_p[y].keys():
            if i == pe_dict[y][-2:]:
                all_vec[y][i] = {}
                target_loc = i
                vec = info_p[y][target_loc].poi_dict[y]['vector']
                all_vec[y][target_loc] = vec            
            
                rg_file.write(y+'\t')
                rg_file.write("\t".join(str(elem) for elem in vec[0]))
                rg_file.write('\n')
    rg_file.close()
    """
    
    
    
    
    ###################################### tune
    #tuneing_data_genfile(filepath, dataset, ts)
    tune_query = tuneing_data_loading_o(filepath, dataset, ts)

    para = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    max_pre = 0
    max_rec = 0
    
    st_alpha = 0
    st_beta = 0
    
    for alpha in para:
        if (1 - alpha) >= 0:
            beta = 1 - alpha
            print('alpha , beta = ' + str(alpha) + ', ' + str(beta))

    
            vi_count = 0
            for u in tune_query:
                vi_count = vi_count+len(tune_query[u])
    
        
            bar4 = progressbar.ProgressBar(widgets=[progressbar.Percentage(),progressbar.Bar()],maxval=vi_count).start()
            p_n4 = 0  
    
            case_count = 0
            item_count = 0
            match_count = 0

            for u in tune_query:
                for cur_poi, (cur_day, cur_time), seq_set, ans_poi, ans_set in tune_query[u]:
            
   
                    context = [(cur_poi, cur_day, cur_time)]

                    top_n = testing_userans(all_pois_freq_dict, user_checked_freq, context, pe_dict, geo_dict, info_p, user_vector, u, filepath, dataset, precision, vec_len, ts, N, a, xmin, alpha, beta)

                    case_count += 1
                    item_count += len(ans_set)
                    match_count += len(set(ans_set) & set(top_n))
           
                time.sleep(0.01)
                p_n4 = p_n4+1
                bar4.update(p_n4)
            bar4.finish()
    
            pre = match_count / float(case_count * N)
            rec = match_count / float(item_count)

            if (pre > max_pre):
                max_pre = pre
                max_rec = rec
                        
                st_alpha = alpha
                st_beta = beta

    
    ########################################################################
    

    #st_alpha, st_beta = 0,0
    case_count = 0
    item_count = 0
    match_count = 0
   
   
    print("testing....")
    
    #testing_data_genfile(filepath, dataset, ts)
    test_query = testing_data_loading_o(filepath, dataset, ts)

    
    vi_count = 0
    for u in test_query:
        vi_count = vi_count+len(test_query[u])
    
    bar = progressbar.ProgressBar(widgets=[progressbar.Percentage(),progressbar.Bar()],maxval=vi_count).start()
    p_n = 0  
    
    for u in test_query:
        for cur_poi, (cur_day, cur_time), seq_set, ans_poi, ans_set in test_query[u]:

            
            #ans_set.add(ans_poi)
            context = [(cur_poi, cur_day, cur_time)]
           
 
            top_n = testing_userans(all_pois_freq_dict, user_checked_freq, context, pe_dict, geo_dict, info_p, user_vector, u, filepath, dataset, precision, vec_len, ts, N, a, xmin, st_alpha, st_beta)
 
            case_count += 1
            item_count += len(ans_set)
            match_count += len(set(ans_set) & set(top_n))
           
            

            time.sleep(0.01)
            p_n = p_n+1
            bar.update(p_n)
    bar.finish()
    
    
    pre = match_count / float(case_count * N)
    rec = match_count / float(item_count)
    
    print(str(st_alpha) + ', ' + str(st_beta) )
    print('Precision a @ ' + str(N) + ' : ' + str(pre))
    print('Recall a @ ' + str(N) + ' : ' + str(rec))
    
 
    res_file = open(filepath + dataset + '_all_res_' + str(ts) + '.txt', 'w')
    res_file.write('poi2vec'+'\n')
    res_file.write(str(st_alpha) + ', ' + str(st_beta) +'\n')
    res_file.write('precision = ' + str(pre) +'\n')
    res_file.write('recall = ' + str(rec) +'\n')
    res_file.close()
