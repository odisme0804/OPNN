import numpy as np
from numpy.linalg import norm
import scipy.stats as st
from collections import defaultdict
from math import *
import progressbar
import time
import multiprocessing as mpc
from operator import itemgetter

def get_default_hparas():
    class hparas():
        pass
    hparas.embedding_path = "./embedding/"
    hparas.successive_time_constrain = 6
    hparas.precision = 20
    hparas.top_N = 10
    hparas.embedding_dim = 300
    hparas.learn_rate = 0.005
    hparas.max_iter = 10
    hparas.epsilon = 0.0001
    hparas.window_size = 3
    hparas.seq_length = 3
    hparas.tree_type = "huffman"
    hparas.model = "cbow"
    hparas.decay = 1
    hparas.inverse = 1
    hparas.merge_bound = 0.8
    # for simularity
    hparas.max_process = 10
    hparas.simu_metric = "max" # [min, max, avg]
    hparas.simu_func = "cos" # [cos, jaccard]

    return hparas

class GeneralTreeNode():
    def __init__(self, value, freq, set_key=-1):
        # common part of leaf node and tree node
        self.freq = freq
        self.left = None  # 
        self.right = None # 
        # value of leaf node  will be the word, and be
        # mid vector in tree node
        self.value = value # the value of word 
        self.path = "" # store the path or huffman code
        self.set_key = set_key

    def __str__(self):
        return 'TreeNode object, value: {v}, freq: {f}, path: {p}'\
            .format(v=self.value,f=self.freq,p=self.path)

class GeneralTree():
    def __init__(self, hparas=None, word_dict=None,
                 word_list=None, user_list=None, user_item_matrix=None,
                 word_mapper=None, inv_word_mapper=None,
                 user_mapper=None, inv_user_mapper=None):
        self.root = None
        if hparas is None:
            self.hparas = get_default_hparas()
        else:
            self.hparas = hparas
        # for simularity
        self.dist_matrix = None
        self.ui_vector = None
        self.resutls = None
        self.word_mapper = word_mapper
        self.inv_word_mapper = inv_word_mapper
        self.user_mapper = user_mapper
        self.inv_user_mapper = inv_user_mapper
        """ 
        word_dict : each element is a dict with key 'word', including: freq, vector, path(code)
        user_item_matrix : Matrix['User']['LOC'] = '1/0' for Simulartiy Tree.  Like:  defaultdict(lambda: defaultdict(int))
        """
 
        if self.hparas.tree_type == "huffman":
            node_list = [GeneralTreeNode(key, value['freq']) for key, value in word_dict.items()]
            self.build_huffman_tree(node_list)
            self.generate_huffman_code(self.root, word_dict)
        elif self.hparas.tree_type == "balance":
            node_list = [GeneralTreeNode(key, value['freq']) for key, value in word_dict.items()]
            self.build_balance_tree(node_list)
            self.generate_huffman_code(self.root, word_dict)
        elif self.hparas.tree_type == "simularity":
            self.cal_distance_matrix(user_item_matrix)
            node_list = [GeneralTreeNode(key, value['freq'], self.word_mapper[key]) for key, value in word_dict.items()]
            self.build_simularity_tree(node_list)
            self.generate_huffman_code(self.root, word_dict)

    def cal_distance_matrix(self, user_item_matrix):
        # initial user-item-matrix
        ui_vector = np.zeros(shape=[len(self.word_mapper), len(self.user_mapper)])
        
        # mark user who visited
        for user in user_item_matrix:
            for word, count in user_item_matrix[user].items():
                if count > 0:
                    wid = self.word_mapper[word]
                    uid = self.user_mapper[user]
                    ui_vector[wid][uid] = 1

        self.ui_vector = ui_vector

        cnttt = 0
        allll = 0
        for i in range(0, len(self.word_mapper)):
            if np.all(ui_vector[i] == 0):
                cnttt += 1
            allll += 1
        print(cnttt, allll)

        tmpp = 0
        # calculate distance between words
        print("compute distance metrix")
        # ver1 
        """ 
        distbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(),progressbar.Bar()],maxval=len(word_list)).start()
        self.dist_matrix = defaultdict(lambda: defaultdict(float))
        for i, word1 in enumerate(word_list):
            time.sleep(0.01)
            distbar.update(i)
            print(i)
            for j, word2 in enumerate(word_list):
                 if i == j: 
                     self.dist_matrix[word1][word2] = self.cal_sim(ui_vector_dict[word1],
                                                                   ui_vector_dict[word2])
                 elif i < j:
                     self.dist_matrix[word1][word2] = self.cal_sim(ui_vector_dict[word1],
                                                               ui_vector_dict[word2])
                     self.dist_matrix[word2][word1] = self.dist_matrix[word1][word2]

                 if self.dist_matrix[word2][word1] == 0.0:
                     tmpp += 1
        distbar.finish()
        print(tmpp)
        self.display_dist_matrix()
        """
        # ver2
        self.dist_matrix = np.zeros(shape=[len(self.word_mapper), len(self.word_mapper)])
 
        manager = mpc.Manager()
        return_dict = manager.dict()

        distbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(),progressbar.Bar()],maxval=self.hparas.max_process).start()
        jobs = []
        for i in range(0, self.hparas.max_process):
            print(i)

            p = mpc.Process(target=self.word_simus, args=(i, self.hparas.max_process, self.hparas.simu_func, return_dict))
            jobs.append(p)
            p.start()


        for i,x in enumerate(jobs):
            distbar.update(i+1)
            x.join()

        distbar.finish()
        """       
        for result, word1 in zip(results, word_list):
            val = result.get()
            for i, word2 in enumerate(word_list):
                #self.dist_matrix[word1][word2] = val[i]
                print(val[i])"""
        #for i, result in enumerate(return_dict.values()):
        #    for j, val in enumerate(result):
        #        self.dist_matrix[i][j] = val
        
        for k,v in return_dict.items():
            #print(k,v)
            self.dist_matrix[k] = v

        # normalize dist matrix
        temp_mat = self.dist_matrix / abs(float(self.dist_matrix.max()))
        self.dist_matrix = temp_mat
        self.display_dist_matrix()

    def word_simus(self, i, total, fun, return_dict):
        for idx in range( len(self.word_mapper) * i//total, len(self.word_mapper)*(i+1)//total):
            ret = []
            for cmpid in range(0, len(self.word_mapper)):
                ret.append(self.cal_sim_test( self.ui_vector[idx],
                                              self.ui_vector[cmpid], fun))
            return_dict[idx] = ret[:]

    def build_simularity_tree(self, node_list):
        word_set = [ x for x in range(0, len(self.dist_matrix))]
        dist_node_pair = []
        for i in range(0, len(self.dist_matrix)):
            for j in range(i+1, len(self.dist_matrix)):
                dist_node_pair.append( [self.dist_matrix[i][j], i, j] )
        #dist_node_pair.sort(key=itemgetter(0), reverse=True)
        dist_node_pair.sort(key=lambda tup: (-tup[0],tup[1],tup[2]))

        merge_pair = []
        for idx, val in enumerate(dist_node_pair):
            d,i,j = val
            temp = i
            while word_set[temp] != temp:
                word_set[temp] = word_set[word_set[temp]]
                temp = word_set[temp]
            i = temp
            temp = j
            while word_set[temp] != temp:
                word_set[temp] = word_set[word_set[temp]]
                temp = word_set[temp]
            j = temp
            if i != j:
                if i > j:
                    i,j = j,i
                merge_pair.append((i,j,d))
                if len(merge_pair) >= len(word_set) - 1:
                    break

            word_set[j] = i
            dist_node_pair[idx] = [d,i,j]

        #merge_pair.sort(key=lambda tup: (-tup[2],tup[0],tup[1]))

        
        modify_pair = []
        last_sim = merge_pair[0][2]
        nodes = set([merge_pair[0][0], merge_pair[0][1]])
        for idx in range(1,len(merge_pair)):
            i,j,d = merge_pair[idx]
            if d != last_sim or (idx == len(merge_pair) - 1) or (i not in nodes and j not in nodes):
                if d == last_sim and idx == len(merge_pair) - 1:
                    nodes.add(i)
                    nodes.add(j) 
                n_list = list(nodes)
                n_list.sort()
                if len(n_list) == 2:
                    modify_pair.append((n_list[0], n_list[1], last_sim)) 
                elif len(n_list) == 3:
                    modify_pair.append((n_list[0], n_list[1], last_sim)) 
                    modify_pair.append((n_list[0], n_list[2], last_sim)) 
                else:
                    while len(n_list) > 1:
                        r, s = n_list[0:2]
                        if r > s:
                            r,s = s,r
                        modify_pair.append((r, s, last_sim))
                        n_list = n_list[2:]
                        n_list.append(min(r,s)) 

                # clean for next

                if d != last_sim and (idx == len(merge_pair) - 1):
                    modify_pair.append((i,j,d))

                # clean for next
                nodes.clear()
                last_sim = d
                nodes.add(i)
                nodes.add(j)
            else:
                nodes.add(i)
                nodes.add(j)

        print("lens:", len(merge_pair), len(modify_pair)) 
        with open('m1.txt', 'w') as the_file:
            for i,j,d in merge_pair:
                the_file.write(str(d)+"\t"+str(i)+"\t"+str(j)+"\n")
                #print(d,i,j)
        print("")
        with open('m2.txt', 'w') as the_file:
            for i,j,d in modify_pair:
                the_file.write(str(d)+"\t"+str(i)+"\t"+str(j)+"\n")
                #print(d,i,j)
        
 
        node_dict = {}
        for node in node_list:
            node_dict[node.set_key] = node
        
        print("build tree ... ")

        treebar = progressbar.ProgressBar(widgets=[progressbar.Percentage(),progressbar.Bar()],maxval=node_list.__len__()).start()
        merge_cnt = 0
        while node_dict.__len__()>1:
            merge_cnt += 1
            treebar.update(merge_cnt)

            nid1, nid2, distance = modify_pair.pop(0)
            if distance < self.hparas.merge_bound:
                break

            top_node = self.merge_by_sim(node_dict[nid1],node_dict[nid2])

            node_dict.pop(nid1, None)
            node_dict.pop(nid2, None)
            node_dict[top_node.set_key] = top_node

        treebar.finish()
        # merge by freq 
        node_list = []
        for key, value in node_dict.items():
            node_list.append(value)

        self.build_huffman_tree(node_list)
        #self.root = node_dict[0]   

    def build_huffman_tree(self, node_list):
        while node_list.__len__()>1:
            i1 = 0  # i1
            i2 = 1  # i2 
            if node_list[i2].freq < node_list[i1].freq :
                [i1,i2] = [i2,i1]
            for i in range(2,node_list.__len__()): # 
                if node_list[i].freq <node_list[i2].freq :
                    i2 = i
                    if node_list[i2].freq < node_list[i1].freq :
                        [i1,i2] = [i2,i1]
            top_node = self.merge_by_freq(node_list[i1],node_list[i2])
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

    def build_balance_tree(self, node_list):
        while node_list.__len__()>1:
            i1 = 0
            i2 = 1
            top_node = self.merge_by_freq(node_list[i1],node_list[i2])
            
            node_list.pop(0)
            node_list.pop(0)
            node_list.append(top_node)

        self.root = node_list[0]

    def generate_huffman_code(self, node, word_dict):
        # # use recursion in this edition
        # if node.left==None and node.right==None :
        #     word = node.value
        #     code = node.Huffman
        #     print(word,code)
        #     word_dict[word]['Huffman'] = code
        #     return -1
        #
        # code = node.Huffman
        # if code==None:
        #     code = ""
        # node.left.Huffman = code + "1"
        # node.right.Huffman = code + "0"
        # self.generate_huffman_code(node.left, word_dict)
        # self.generate_huffman_code(node.right, word_dict)

        # use stack but not recursion in this edition
        total = 0.0
        count = 0
        stack = [self.root]
        while (stack.__len__()>0):
            node = stack.pop()
            # go along left tree
            while node.left or node.right :
                path = node.path
                node.left.path = path + "1"
                node.right.path = path + "0"
                stack.append(node.right)
                node = node.left
            word = node.value
            path = node.path
            # print(word,'\t',code.__len__(),'\t',node.possibility)
            word_dict[word]['path'] = path
            if not node.left and not  node.right: # leaf
                #print(word, path)
                total += len(path)
                count += 1
        print("avg len:", total/count)

    def merge_by_freq(self,node1,node2):
        top_pos = node1.freq + node2.freq
        top_node = GeneralTreeNode(np.zeros([1,self.hparas.embedding_dim]), top_pos)
        #top_node = GeneralTreeNode(st.truncnorm.rvs(-1, 1, size=[1,vec_len])), top_pos)
        if node1.freq >= node2.freq :
            top_node.left = node1
            top_node.right = node2
        else:
            top_node.left = node2
            top_node.right = node1
        return top_node

    def merge_by_sim(self,node1,node2):
        top_pos = node1.freq + node2.freq
        top_node = GeneralTreeNode(np.zeros([1,self.hparas.embedding_dim]), top_pos, min(node1.set_key, node2.set_key))
        if node1.freq >= node2.freq :
            top_node.left = node1
            top_node.right = node2
        else:
            top_node.left = node2
            top_node.right = node1
        return top_node

    def test(self):
        print("hello")

    def display_dist_matrix(self):
        """
        for wid1 in range(0, len(self.word_mapper)):
            for wid2 in range(0, len(self.word_mapper)):
                print(self.dist_matrix[wid1][wid2], end=' ')
            print("")
        print("")
        """
        print(self.dist_matrix)
        #print(self.word_mapper)

    def cal_sim(self, v1, v2):
        if self.hparas.simu_func == "cos":
            return (np.inner(v1,v2)/(norm(v1)*norm(v2)))[0][0] if norm(v1)*norm(v2) > 0 else 0.0
        elif self.hparas.simu_func == "mht":
            return -1 * sum(abs(a-b) for a,b in zip(v1[0], v2[0]))
        elif self.hparas.simu_func == "euc":
            return -1 * sqrt(sum(pow(a-b, 2) for a,b in zip(v1[0], v2[0])))
        elif self.hparas.simu_func == "jaccard":
            q = r = s = 0
            for i in range(0, len(v1[0])):
                if v1[0][i] == 1 and v2[0][i] == 1:
                    q += 1
                elif v1[0][i] == 1:
                    r += 1
                elif v2[0][i] == 1:
                    s += 1
            return q/(q + r + s)


    def cal_sim_test(self, v1, v2, fun):
        if fun == "cos":
            return (np.inner(v1,v2)/(norm(v1)*norm(v2))) if norm(v1)*norm(v2) > 0 else 0
        elif fun == "mht":
            return -1 * np.sum(np.absolute(v1 - v2))
            #sum(abs(a-b) for a,b in zip(v1, v2))
        elif fun == "euc":
            return -1 * norm(v1-v2)
            #sqrt(sum(pow(a-b, 2) for a,b in zip(v1, v2)))
        elif fun == "jaccard":
            return np.sum(np.minimum.reduce([v1.v2])) / np.sum(np.maximum.reduce([v1.v2]))
            """
            q = r = s = 0
            for i in range(0, len(v1)):
                if v1[i] == 1 and v2[i] == 1:
                    q += 1
                elif v1[0][i] == 1:
                    r += 1
                elif v2[0][i] == 1:
                    s += 1
            return q/(q + r + s)
            """
