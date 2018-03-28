import math

from Mylibs.GeneralTree import GeneralTree, get_default_hparas
from Mylibs.Tools import *
import progressbar
import numpy as np
import time
from sklearn import preprocessing
from collections import OrderedDict
import multiprocessing as mpc

class Word2Vec():
    def __init__(self, hparas=None):
        if hparas is None:
            self.hparas = get_default_hparas()
        else:
            self.hparas = hparas
 
        self.user_dict = None
        self.word_dict = None  # each element is a dict, including: freq, vector, path
        self.tree = None    # the object of HuffmanTree
        self.user_mapper = None
        self.word_mapper = None
        self.inv_user_mapper = None
        self.inv_word_mapper = None
        self.condidate_dict = {}

    def initial_word_dict(self, word_freq):
        # the input is dict[word] = freq , output is dict[word] = {freq:"", vector:"", path:""}
        if not isinstance(word_freq,dict):
            raise valueerror('the word freq info should be a dict')

        word_dict = {}
        for word, freq in word_freq.items():
            temp_dict = dict(
                word = word,
                freq = freq * self.hparas.inverse,
                vector = np.random.random([1,self.hparas.embedding_dim]),
                path = None
            )
            word_dict[word] = temp_dict

        self.word_dict = word_dict
        
        self.word_mapper = {}
        self.inv_word_mapper = {}
        for i, word in enumerate(self.word_dict):
            self.word_mapper[word] = i
            self.inv_word_mapper[i] = word

    def initial_user_dict(self, user_list):
        # the input is dict[word] = fred , output is dict[word] = {freq:"", vector:"", path:""}
        if not isinstance(user_list,list):
            raise valueerror('the user list should be a list of uid')

        user_dict = {}
        for uid in user_list:
            user_dict[uid] = np.random.random([1,self.hparas.embedding_dim])

        self.user_dict = user_dict
        
        self.user_mapper = {}
        self.inv_user_mapper = {}
        for i, user in enumerate(self.user_dict):
            self.user_mapper[user] = i
            self.inv_user_mapper[i] = user

    def train(self, train_dict):
        # train_dict : Dict[user] = [(context, current, after), ()]
        if self.word_dict == None:
            raise ValueError('Need initial word_dict')
        if self.tree == None:
            if self.hparas.tree_type == "huffman":
                self.tree = GeneralTree(self.hparas, self.word_dict)
             
        print("Tree has been built, start to training...")

        # start to train word vector
        before = (self.hparas.window_size - 1) >> 1
        after  =  self.hparas.window_size - 1 - before

        if self.hparas.model=='cbow':
            method = self.__Deal_Gram_CBOW
        else:
            method = self.__Deal_Gram_SkipGram

        total_query = len(train_dict) * self.hparas.max_iter
        current_count = 0
        pgbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(),progressbar.Bar()],maxval=total_query).start()
        
        for _ in range(0, self.hparas.max_iter):
            for user in train_dict:
                for before, current, after in train_dict[user]:
                    if not before:
                        continue

                    context = list(set(before))
                    gd_temp = list(set(after)) # not user current
                    method( current, context, gd_temp) 
                time.sleep(0.001) 
                current_count += 1
                pgbar.update(current_count)
        pgbar.finish()
        time.sleep(0.1)
        print('word vector has been generated')


    def train_all(self, context_dict, nocontext_dict, user_item_matrix=None):
        # Dict format : Dict[user] = [(context, current, after), ()]
        if self.word_dict == None:
            raise ValueError('Need initial word_dict')
        if self.user_dict == None:
            raise ValueError('Need initial user_dict')
        if self.tree == None:
            if self.hparas.tree_type == "huffman":
                self.tree = GeneralTree(self.hparas, self.word_dict)
            elif self.hparas.tree_type == "balance":
                self.tree = GeneralTree(self.hparas, self.word_dict)
            elif self.hparas.tree_type == "simularity":
                self.tree = GeneralTree(self.hparas, self.word_dict,
                                        list(self.word_dict.keys()),
                                        list(self.user_dict.keys()),
                                        user_item_matrix,
                                        self.word_mapper, self.inv_word_mapper,
                                        self.user_mapper, self.inv_user_mapper,
                                        )

             
        print("Tree has been built, start to training...")

        # start to train word vector
        before = (self.hparas.window_size - 1) >> 1
        after  =  self.hparas.window_size - 1 - before

        if self.hparas.model=='cbow':
            method = self.__Deal_Gram_CBOW
        else:
            method = self.__Deal_Gram_SkipGram

        total_query = len(context_dict) * self.hparas.max_iter
        current_count = 0
        pgbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(),progressbar.Bar()],maxval=total_query).start()
        
        for _ in range(0, self.hparas.max_iter):
            for user in context_dict:
                for before, current, after in context_dict[user]:
                    if not before:
                        print("plz check gen_seq func")
                        continue

                    #context = []
                    context = list(OrderedDict.fromkeys(before).keys())
                    #context = list(set(before))
                    gd_temp = list(set(after)) # not user current
                    method( current, context[max(0, context.__len__() - self.hparas.window_size): ], gd_temp)

                # calculate user vector
                for before, current, after in nocontext_dict[user]:
                    if before:
                        print("plz check gen_seq func")
                        continue

                    gd_temp = list(set(after)) # not user current
                    self.__Cal_User_Vector( current, user, gd_temp)

                time.sleep(0.001) 
                current_count += 1
                pgbar.update(current_count)
            self.hparas.learn_rate *= self.hparas.decay

        pgbar.finish()
        time.sleep(0.1)
        print('word vector has been generated')

    def __Cal_User_Vector(self, word, user, gd):

        if not self.word_dict.__contains__(word):
            print(str(word)+" not in dict")
            return

        word_path = self.word_dict[word]['path']
        user_vector = np.zeros([1,self.hparas.embedding_dim])
            
        if self.user_dict.__contains__(user):
            user_vector = self.user_dict[user]
        else:
            print(str(user) + " not in dict")
            return

        #if gram_word_list.__len__()==0:
        #    return

        e = self.__GoAlong_Huffman(word_path, user_vector, self.tree.root)

        self.user_dict[user] += e
        self.user_dict[user] = preprocessing.normalize(self.user_dict[user])

    def __Deal_Gram_CBOW(self, word, gram_word_list, gd):

        if not self.word_dict.__contains__(word):
            print(str(word)+" not in dict")
            return

        word_path = self.word_dict[word]['path']
        gram_vector_sum = np.zeros([1,self.hparas.embedding_dim])
        for  i in range(gram_word_list.__len__())[::-1]:
            item = gram_word_list[i]
            if self.word_dict.__contains__(item):
                gram_vector_sum += self.word_dict[item]['vector']
            else:
                gram_word_list.pop(i)
                print(str(word) + " not in dict")

        #if gram_word_list.__len__()==0:
        #    return

        e = self.__GoAlong_Huffman(word_path, gram_vector_sum, self.tree.root)

        for item in gram_word_list:
            self.word_dict[item]['vector'] += e
            self.word_dict[item]['vector'] = preprocessing.normalize(self.word_dict[item]['vector'])

    def __Deal_Gram_SkipGram(self, word, gram_word_list, gd):

        if not self.word_dict.__contains__(word):
            print(str(word)+" not in dict")
            return

        word_vector = self.word_dict[word]['vector']
        for i in range(gram_word_list.__len__())[::-1]:
            if not self.word_dict.__contains__(gram_word_list[i]):
                print(str(gram_word_list[i])+" not in dict")
                gram_word_list.pop(i)

        #if gram_word_list.__len__()==0:
        #    return

        for u in gram_word_list:
            u_path = self.word_dict[u]['path']
            e = self.__GoAlong_Huffman(u_path, word_vector, self.tree.root)
            self.word_dict[word]['vector'] += e
            self.word_dict[word]['vector'] = preprocessing.normalize(self.word_dict[word]['vector'])

    def __GoAlong_Huffman(self, word_path, input_vector, root):

        node = root
        e = np.zeros([1, self.hparas.embedding_dim])
        for level in range(word_path.__len__()):
            path_charat = word_path[level]
            q = self.__Sigmoid(input_vector.dot(node.value.T))
            grad = self.hparas.learn_rate * ( 1 - int(path_charat) - q)
            e += grad * node.value
            node.value += grad * input_vector
            node.value = preprocessing.normalize(node.value)
            if path_charat == '0':
                node = node.right
            else:
                node = node.left
        return e

    def __Sigmoid(self,value):
        return 1/(1+math.exp(-value))

    def cal_probability(self, a, b):
        return self.__Sigmoid(a.dot(b.T)[0][0])

    def get_top_k(self, user, query, poi_info, K=10, A=1, B=1):
        before, current, after = query
        if current in self.condidate_dict:
            candidate_poi = self.condidate_dict[current]
        else:
            candidate_poi = dist_filter(current, poi_info, 10)
            self.condidate_dict[current] = candidate_poi[:]
        
        con_list = []
        for poi in before:
            con_list.append(poi)
        con_list.append(current)
        context = list(OrderedDict.fromkeys(con_list).keys())
        #context = [context.append(item) for item in con_list if item not in context]
        
        #context = con_list 
        #context = list(set(con_list + before))

        context_vector_sum = np.zeros([1,self.hparas.embedding_dim])
        for cnt, i in enumerate(range(context.__len__())[::-1]):
            if cnt >= self.hparas.window_size:
                break

            item = context[i]
            if self.word_dict.__contains__(item):
                context_vector_sum += self.word_dict[item]['vector']
            else:
                context.pop(i)
                print(str(word) + " not in dict")

        poi_user_score_dict = {}
        poi_context_score_dict = {}
        for poi in candidate_poi:
            poi_user_score_dict[poi] = np.dot(self.user_dict[user], self.word_dict[poi]['vector'].T)[0][0]
            poi_context_score_dict[poi] = np.dot(context_vector_sum, self.word_dict[poi]['vector'].T)[0][0]
        
        sum_pu_score = cal_normalize(poi_user_score_dict)
        sum_pc_score = cal_normalize(poi_context_score_dict)

        poi_score = {} 
        for poi in candidate_poi:
            poi_score[poi] = A * sum_pu_score[poi] + B * sum_pc_score[poi]
        
        scores = list(sorted(poi_score, key=poi_score.__getitem__, reverse=True))
    
        return scores[:K]

    def save_embedding(self):
        np.save(self.hparas.embedding_path + 'word_vector.npy', self.word_dict)
        np.save(self.hparas.embedding_path + 'user_vector.npy', self.user_dict)
        np.save(self.hparas.embedding_path + 'word_mapper.npy', self.word_mapper)
        np.save(self.hparas.embedding_path + 'user_mapper.npy', self.user_mapper)

    def load_embedding(self):
        self.word_dict = np.load(self.hparas.embedding_path + 'word_vector.npy').item()
        self.user_dict = np.load(self.hparas.embedding_path + 'user_vector.npy').item()
        self.word_mapper = np.load(self.hparas.embedding_path + 'word_mapper.npy').item()
        self.user_mapper = np.load(self.hparas.embedding_path + 'user_mapper.npy').item()

    def get_embedding_matrix(self):
        word_mat = np.zeros([len(self.word_dict), self.hparas.embedding_dim])
        user_mat = np.zeros([len(self.user_dict), self.hparas.embedding_dim])
        for k,v in self.word_dict.items():
            word_mat[ self.word_mapper[k]] = v
        for k,v in self.user_dict.items():
            user_mat[ self.user_mapper[k]] = v
        return word_mat, user_mat

