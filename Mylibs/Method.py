import numpy as np
from collections import defaultdict
import datetime
import h5py
import os
import time
import random
from os import listdir
from os.path import isfile, join
from shutil import copyfile
from keras.models import Model
from keras.layers import Dense, Embedding, Lambda, LSTM, Input, Reshape, Activation, RepeatVector, Maximum
from keras.layers import Add, Concatenate, Flatten, MaxoutDense, TimeDistributed, Multiply
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import regularizers
from keras import constraints
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras import metrics
from Mylibs.Tool import *
import multiprocessing as mpc
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering as ac

def get_NN_paras():
    class paras():
        pass
    paras.seq_len = 3
    paras.learn_rate = 0.0005
    paras.max_iter = 10
    paras.batch_size = 512
    paras.epoch = 20
    paras.hidden_layer = 3
    paras.hidden_dim = 300
    paras.drop_rate = 0.2
    paras.train_poi = False
    paras.train_user = False
    paras.max_process = 20
    paras.nb_cluster = 3
    paras.user_cluster = 3
    paras.augment_rate = 0.05

    return paras


class OPNN:
    def __init__(self, paras=None):
        if paras is None:
            self.paras = get_NN_paras()
        else:
            self.paras = paras
        self.poi2idx = None
        self.idx2poi = None
        self.user2idx = None
        self.idx2user = None
        self.poi_mat = None
        self.user_mat = None
        self.model = None
        self.dist_mat = None
        self.train_visit = defaultdict(set)
        self.train_tune_visit = defaultdict(set)
        self.poi_info = None
        self.centers = None
        self.u_labels = None
        self.u_centers = None
        self.ug_mat = None
        self.x_train, self.u_train, self.y_train, self.d_train = None, None, None, None
        self.x_val, self.u_val, self.y_val, self.d_val = None, None, None, None
        self.x_tune, self.u_tune, self.y_tune, self.d_tune = None, None, None, None
        self.t_train, self.t_tune, self.t_val = None, None, None

    def load_embedding(self, path):
        # load pre-train embedding vectors
        word_dict = np.load(path + 'word_vector.npy').item()
        user_dict = np.load(path + 'user_vector.npy').item()
        # add no checkin vector
        word_dict["NO_LOC"] = {'vector' : np.zeros([1, 300])}
        user_dict["NO_USR"] = np.zeros([1, 300])

        self.poi2idx = dict((p, i) for i, p in enumerate(word_dict))
        self.idx2poi = dict((i, p) for i, p in enumerate(word_dict))

        self.user2idx = dict((u, i) for i, u in enumerate(user_dict))
        self.idx2user = dict((i, u) for i, u in enumerate(user_dict))

        # create words' and users' vector maxtrix
        self.poi_mat  = np.zeros([ len(word_dict), 300 ]) # NO_LOC is the largest idx
        self.user_mat = np.zeros([ len(user_dict), 300 ])
        for k,v in word_dict.items():
            self.poi_mat[ self.poi2idx[k] ] = v['vector']

        for k,v in user_dict.items():
            self.user_mat[ self.user2idx[k] ] = v

        print(self.poi_mat.shape)
        print(self.user_mat.shape)

    def compute_user_embedding_v2(self):
        if self.paras.user_cluster <= 1:
            self.u_labels = np.zeros([ self.user_mat.shape[0] + 1])
            self.ug_mat = np.zeros( [ self.paras.user_cluster, self.paras.hidden_dim ] )
            return

        user_check = np.zeros( [self.user_mat.shape[0], self.poi_mat.shape[0]] )
        for i in range(0, len(self.x_train)):
            u   = self.u_train[i][-1]
            con = self.x_train[i][-1]
            fut = self.y_train[i]
            user_check[u][con] += 1
        
        kmeans = KMeans(n_clusters=self.paras.user_cluster)
        kmeans.fit(user_check)
        self.u_labels = np.array(kmeans.labels_)
        self.u_centers = np.array(kmeans.cluster_centers_)
        ug_ebd = np.zeros( [ self.paras.user_cluster, self.user_mat.shape[1] ] )

        for i in range(0, self.paras.user_cluster):
            ug_ebd[i] = np.sum( np.multiply(self.u_centers[i].reshape([1, -1]), self.poi_mat.T), axis=1 )
        self.ug_mat = ug_ebd
        #self.ug_mat = np.array(normalize(self.ug_mat, axis=0, norm='l1'))
        print('ug_mat_shape:', self.ug_mat.shape)

    def compute_user_embedding_v3(self):
        if self.paras.user_cluster <= 1:
            self.u_labels = np.zeros([ self.user_mat.shape[0] + 1])
            self.ug_mat = np.zeros( [ self.paras.user_cluster, self.paras.hidden_dim ] )
            return
        user_check = np.zeros( [self.user_mat.shape[0], self.poi_mat.shape[0]] )
        for i in range(0, len(self.x_train)):
            u   = self.u_train[i][-1]
            con = self.x_train[i][-1]
            fut = self.y_train[i]
            user_check[u][con] = 1
        
        AC = ac(n_clusters=self.paras.user_cluster, affinity='cosine', linkage='complete')
        labels = AC.fit_predict(user_check)
        self.u_labels = np.array(labels_)
        print('label_shape:', self.u_labels.shape)

    def compute_user_embedding(self):
      
        #self.user_mat = np.array(normalize(self.user_mat, axis=0, norm='l1'))
        new_u = np.zeros( self.user_mat.shape )
        new_c = np.ones( [self.user_mat.shape[0], 1] )
        user_check = np.zeros( [self.user_mat.shape[0], self.poi_mat.shape[0]] )
        check_norm = np.zeros( [self.user_mat.shape[0], 1] )
        u_sets = defaultdict(set)
        """
        current_day = datetime.datetime(2000,1,1,0,0,0)
        for ckday in self.d_train:
            if ckday > current_day:
                current_day = ckday
        print(current_day)
        """
        alpha = 4/4.

        for i in range(0, len(self.x_train)):
            u   = self.u_train[i][-1]
            con = self.x_train[i][-1]
            fut = self.y_train[i]

            user_check[u][con] = 1

        user_check = pow(user_check, alpha)
        where_are_NaNs = np.isnan(user_check)
        user_check[where_are_NaNs] = 0
        user_check[user_check == np.inf] = 0
        user_check[user_check == -np.inf] = 0
        check_norm = np.sum(user_check, axis=-1)

        for i in range(0, len(self.x_train)):
            u   = self.u_train[i][-1]
            con = self.x_train[i][-1]
            fut = self.y_train[i]

            if user_check[u][con] == 0:
                continue

            new_u[u] += self.poi_mat[con] * user_check[u][con] / check_norm[u]
            #new_c[u] = 1.0
            user_check[u][con] = 0

        """
        for u in u_sets.keys():
            for c in list(u_sets[u]):
                new_u[u] += self.poi_mat[c]
                new_c[u] += 1.0
        
            
            new_u[u] -= self.poi_mat[con]
            new_u[u] += self.poi_mat[fut]
            new_c[u] += 1.0
        """

        #with np.errstate(divide='ignore', invalid='ignore'):
        #    new_u /= new_c

        where_are_NaNs = np.isnan(new_u)
        new_u[where_are_NaNs] = 0
        #new_u = np.array(normalize(new_u, axis=0, norm='l1'))
        self.user_mat = new_u

    def build_model(self):
        poi_ebd_layer = Embedding(self.poi_mat.shape[0], self.poi_mat.shape[1],
                                  weights=[self.poi_mat],
                                  input_length=self.paras.seq_len,
                                  trainable=self.paras.train_poi)
                                  #embeddings_regularizer=regularizers.l2(0.01))
        user_ebd_layer = Embedding(self.user_mat.shape[0], self.user_mat.shape[1],
                                   weights=[ np.zeros(self.user_mat.shape) ],
                                   #weights=[self.user_mat],
                                   input_length=self.paras.seq_len,
                                   trainable=self.paras.train_user)
                                   #embeddings_regularizer=regularizers.l2(0.01))

        poi_input = Input(shape=(self.paras.seq_len,), name='poi_input')
        poi = poi_ebd_layer(poi_input)
        user_input = Input(shape=(self.paras.seq_len,), name='user_input')
        user = user_ebd_layer(user_input)

        time_input = Input(shape=(1,), name='time_input')
        weekday_ebd = Embedding(48, self.paras.hidden_dim,
                                 input_length=1,
                                 trainable=True)
        wkd = weekday_ebd(time_input)
        wkd = Flatten()(wkd)
        wkd = RepeatVector(self.paras.seq_len)(wkd)

        # how to merge 2 vec
        merged = Concatenate()([poi, user, wkd])
        merged = BatchNormalization()(merged)
#        merged = BatchNormalization(gamma_constraint=constraints.max_norm(1), gamma_regularizer=regularizers.l2(0.01),
#                                    beta_constraint=constraints.max_norm(1), beta_regularizer=regularizers.l2(0.01))(merged)


        x = LSTM(self.paras.hidden_dim, return_sequences=False, dropout=self.paras.drop_rate, recurrent_dropout=0.5)(merged)

        # add new input
        """
        trav_input = Input(shape=(self.paras.seq_len,), name='trav_input')
        mobi = BatchNormalization()(trav_input)
        mobi = Dense(self.paras.seq_len*2, activation='relu', kernel_regularizer=regularizers.l1(0.01))(mobi)
        mobi = Dense(1, activation='sigmoid')(mobi)
        mobi = RepeatVector(self.dist_mat.shape[0])(mobi)
        """
        grid_input = Input(shape=(1,), name='grid_input')
        grid_bias_layer = Embedding(self.paras.nb_cluster, 1,
                                    weights=[ np.zeros([self.paras.nb_cluster, 1]) ],
                                    input_length=1,
                                    trainable=True)
        grid_b = grid_bias_layer(grid_input)
        grid_b = Flatten()(grid_b)
        grid_b = RepeatVector(self.dist_mat.shape[0])(grid_b)


        dist_input = Input(shape=(1,), name='dist_input')
        poi_dist_layer_1 = Embedding(self.dist_mat.shape[0], self.dist_mat.shape[1],
                                     weights=[ np.log(self.dist_mat + 0.000001) ],
                                     input_length=1,
                                     trainable=False)


        ug_input = Input(shape=(1,), name='user_group_input')
        ug_ebd_layer = Embedding(self.ug_mat.shape[0], self.ug_mat.shape[1],
                                 weights=[ np.zeros(self.ug_mat.shape) ],
                                 input_length=1,
                                 #embeddings_regularizer=regularizers.l2(0.1),
                                 trainable=False)
        ug = ug_ebd_layer(ug_input)
        ug = Flatten()(ug)
        ug = BatchNormalization()(ug)

        dist_layer = poi_dist_layer_1(dist_input)
        #dist_layer_2 = poi_dist_layer_2(dist_input)
        #dist_layer_1 = Flatten()(dist_layer_1)
        #dist_layer_2 = Flatten()(dist_layer_2)
        dist_layer = Reshape((self.dist_mat.shape[0],1))(dist_layer)
        #dist_layer_2 = Reshape((self.dist_mat.shape[0],1))(dist_layer_2)
        #dist_layer = Concatenate()([dist_layer_1, dist_layer_2])

        #time_input = Input(shape=(1,), name='time_input')

        #time_trans = Embedding(24, self.paras.hidden_dim/30,
        #                       weights=[ np.ones([24, 10]) ],
        #                       input_length=1,trainable=True)
        #time_influence = time_trans(time_input)
        #time_influence = Multiply()([user_t, time_influence])
        #time_influence = Dense(1, activation="sigmoid")(time_influence)
        #time_influence = Flatten()(time_influence)
        #time_influence = RepeatVector(self.dist_mat.shape[0])(time_influence)

        #time_one_hot = Flatten()(time_one_hot)
        #time_influence = Dense(self.paras.hidden_dim, activation="tanh")(time_one_hot)
        #time_influence = Flatten()(time_one_hot) 
        #x = Concatenate()([x, time_influence])

        time_w_layer = Embedding(48 * self.paras.nb_cluster, 1, weights=[ -0.7 * np.ones([48 * self.paras.nb_cluster, 1]) ],
                                 input_length=1,
                                 trainable=True)
        time_b_layer = Embedding(48 * self.paras.nb_cluster, 1, weights=[ np.ones([48 * self.paras.nb_cluster, 1]) ],
                                 input_length=1,
                                 trainable=True)

        time_ebd_w = time_w_layer(time_input)
        time_ebd_b = time_b_layer(time_input)
        time_ebd_w = Flatten()(time_ebd_w)
        time_ebd_b = Flatten()(time_ebd_b)
        time_ebd_w = RepeatVector(self.dist_mat.shape[0])(time_ebd_w)
        time_ebd_b = RepeatVector(self.dist_mat.shape[0])(time_ebd_b)

        #time_ebd_w = Add()([time_ebd_w, mobi]) # new add
        st_influence = Multiply()([dist_layer, time_ebd_w])
        st_influence = Add()([st_influence, time_ebd_b])
        st_influence = Add()([st_influence, grid_b])
        #st_influence = Add()([st_influence, mobi]) # new add
       
        #st_influence = Multiply()([st_influence, time_influence])
        st_influence = Flatten()(st_influence)
        st_influence = Activation("sigmoid")(st_influence)

        #merge_st = concatenate([dist_layer, time_input])
        #merge_st = Dense(self.poi_mat.shape[0], activation='tanh')(merge_st)
        #merge_st = Dense(self.poi_mat.shape[0], activation='relu')(merge_st)

        #mm = MaxoutDense(30, nb_feature=3)(merge_st)

        #x = Flatten()(x)

        # add maxout
        #x = MaxoutDense(self.paras.hidden_dim, nb_feature=2)(x)
 
        #x = concatenate([x, mm])
        #output = Dense(self.poi_mat.shape[0])(x)
        #output = Multiply()([output, st_influence])
        x = Concatenate()([x, ug])
        output = Dense(self.poi_mat.shape[0], activation='softmax')(x)
        output = Multiply()([output, st_influence])
        #x = Dense(self.poi_mat.shape[0])(x)
        #x = BatchNormalization()(x)
        #output = Activation("softmax")(x)
        model = Model(inputs=[poi_input, user_input, dist_input, ug_input, time_input, grid_input], outputs=output)
        #model = Model(inputs=[poi_input, user_input, dist_input, single_input, time_input], outputs=output)
        #model = Model(inputs=[poi_input, user_input, dist_input, time_input], outputs=output)
        #model = Model(inputs=[poi_input, user_input], outputs=output)
        adam = optimizers.Adam(lr=self.paras.learn_rate)
        model.compile(loss='categorical_crossentropy', optimizer=adam,
                      metrics=['accuracy', metrics.top_k_categorical_accuracy])
        self.model = model

    def build_dataset(self, train_file, tune_file, test_file):
        users = []
        contexts = []
        answers = []
        dates = []
        travs = []
        with open(train_file, 'r') as fin:
            for row in fin:
                data = row.strip().split('\t')
                u, con, datestr, gd = data[0], data[1], data[2], data[3]
                context = [ self.poi2idx[p] for p in con.split()]
                user = [self.user2idx[u]] * len(context)

                #if len(context) >= self.paras.seq_len and random.random() < self.paras.augment_rate:
                #    context[-2], context[-3] = context[-3], context[-2]

                while len(context) < self.paras.seq_len:
                    context.insert(0, self.poi2idx["NO_LOC"])
                    user.insert(0, self.user2idx["NO_USR"])
                users.append(user[-1 * self.paras.seq_len : ])
                contexts.append(context[-1 * self.paras.seq_len : ])
                future = [ self.poi2idx[p] for p in gd.split()]
                answers.append(future[0])
                checkin_time = datetime.datetime.strptime(datestr,"%Y-%m-%d %H:%M:%S")
                dates.append(checkin_time)
                # dev
                trav = [-13.8]
                for i in range(1, len(context)):
                    trav.append(np.log(self.dist_mat[ context[i-1] ][ context[i] ] + 0.000001))
                travs.append(trav[-1 * self.paras.seq_len : ])
                # update user_visited
                self.train_visit[self.user2idx[u]] |= set(context + future)
                self.train_tune_visit[self.user2idx[u]] |= set(context + future)

        contexts = np.array(contexts)
        users = np.array(users)
        dates = np.array(dates)
        answers = np.array(answers)
        travs = np.array(travs)
        # shuffle
        indices = np.arange(contexts.shape[0])
        np.random.shuffle(indices)
        self.x_train = contexts[indices]
        self.u_train = users[indices]
        self.d_train = dates[indices]
        self.y_train = answers[indices]
        self.t_train = travs[indices]
        # for check
        for i in range(self.paras.seq_len):
            print(self.idx2user[users[5][i]], self.idx2poi[contexts[5][i]], dates[0], self.idx2poi[answers[5]], travs[5][i])

        # load tuning
        users = []
        contexts = []
        answers = []
        dates = []
        travs = []
        with open(tune_file, 'r') as fin:
            for row in fin:
                data = row.strip().split('\t')
                u, con, datestr, gd = data[0], data[1], data[2], data[3]
                context = [ self.poi2idx[p] for p in con.split()]
                user = [self.user2idx[u]] * len(context)
                while len(context) < self.paras.seq_len:
                    context.insert(0, self.poi2idx["NO_LOC"])
                    user.insert(0, self.user2idx["NO_USR"])
                users.append(user[-1 * self.paras.seq_len : ])
                contexts.append(context[-1 * self.paras.seq_len : ])
                future = [ self.poi2idx[p] for p in gd.split()]
                answers.append(future)
                checkin_time = datetime.datetime.strptime(datestr,"%Y-%m-%d %H:%M:%S")
                dates.append(checkin_time)
                # dev
                trav = [-13.8]
                for i in range(1, len(context)):
                    trav.append(np.log(self.dist_mat[ context[i-1] ][ context[i] ] + 0.000001))
                travs.append(trav[-1 * self.paras.seq_len : ])
                # update user_visited
                self.train_tune_visit[self.user2idx[u]] |= set(context + future)


        self.x_tune = np.array(contexts)
        self.u_tune = np.array(users)
        self.d_tune = np.array(dates)
        self.y_tune = np.array(answers)
        self.t_tune = np.array(travs)
        # load testing
        users = []
        contexts = []
        answers = []
        dates = []
        travs = []
        with open(test_file, 'r') as fin:
            for row in fin:
                data = row.strip().split('\t')
                u, con, datestr, gd = data[0], data[1], data[2], data[3]
                context = [ self.poi2idx[p] for p in con.split()]
                user = [self.user2idx[u]] * len(context)
                while len(context) < self.paras.seq_len:
                    context.insert(0, self.poi2idx["NO_LOC"])
                    user.insert(0, self.user2idx["NO_USR"])
                users.append(user[-1 * self.paras.seq_len : ])
                contexts.append(context[-1 * self.paras.seq_len : ])
                future = [ self.poi2idx[p] for p in gd.split()]
                answers.append(future)
                checkin_time = datetime.datetime.strptime(datestr,"%Y-%m-%d %H:%M:%S")
                dates.append(checkin_time)
                # dev
                trav = [-13.8]
                for i in range(1, len(context)):
                    trav.append(np.log(self.dist_mat[ context[i-1] ][ context[i] ] + 0.000001))
                travs.append(trav[-1 * self.paras.seq_len : ])

        self.x_val = np.array(contexts)
        self.u_val = np.array(users)
        self.d_val = np.array(dates)
        self.y_val = np.array(answers)
        self.t_val = np.array(travs)

    def __generator(self, V):
        batch = self.paras.batch_size
        while True:
            # shuffle
            indices = np.arange(self.x_train.shape[0])
            np.random.shuffle(indices)
            poi    = self.x_train[indices]
            user   = self.u_train[indices]
            date   = self.d_train[indices]
            labels = self.y_train[indices]
            trav   = self.t_train[indices]
            for i in range(len(poi) / batch):
                X = poi[i * batch:(i + 1) * batch]
                for j in range(0, batch):
                    if X[j][0] != self.poi2idx["NO_LOC"] and random.random() < self.paras.augment_rate:
                        X[j][-2], X[j][-3] = X[j][-3], X[j][-2]
                U = user[i * batch:(i + 1) * batch]
                P = X[:,-1]
                T = U[:,-1]
                #P = [ [self.poi_info[ self.idx2poi[pp]][1], self.poi_info[ self.idx2poi[pp] ][2]] for pp in P ]
                #P = np.array(P)
                y = to_categorical(labels[i * batch:(i + 1) * batch], V)
                temp = [x.hour for x in date[i * batch:(i + 1) * batch] ]
                temp = np.array(temp)
                is_week = np.array([x.weekday()/5 for x in date[i * batch:(i + 1) * batch] ]) 
                Loc = [ self.centers[x] for x in P ]
                Loc = np.array(Loc)
                Uc = [ self.u_labels[x] for x in T ] # user cluter label
                Uc = np.array(Uc)
                #d = to_categorical(temp, 24) 
                #yield [X, U], y
                #yield [X, U, P, temp + 24 * is_week], y
                #yield [X, U, P, T, temp + 24 * is_week], y
                #yield [X, U, P, T, temp + 24 * is_week + 48 * Loc], y
                #yield [X, U, P, T, temp + 24 * is_week,  Loc], y
                yield [X, U, P, Uc, temp + 24 * is_week,  Loc], y

    def load_dist_matrix(self, fp, prefix=""):
        if os.path.isfile(fp + prefix + '_dist.npy') and os.path.isfile(fp + prefix + '_km.npy'):
            self.dist_mat = np.load(fp + prefix + '_dist.npy')
            self.poi_info = np.load(fp + prefix + '_km.npy').item()
        else:
            self.dist_mat, self.poi_info = get_dist_matrix(fp, self.poi2idx)
        np.save(fp + prefix + '_dist.npy', self.dist_mat)
        np.save(fp + prefix + '_km.npy', self.poi_info)

        print("load dist matrix fin.")

    def compute_center(self):
        X = np.zeros([len(self.poi_info) + 1,2])
        X[ self.poi2idx["NO_LOC"] ] = [10000,0]
        for key, tup in self.poi_info.items():
            X[ self.poi2idx[key] ] = [tup[1], tup[2]]

        if self.paras.nb_cluster > 1:
            kmeans = KMeans(n_clusters=self.paras.nb_cluster)
            kmeans.fit(X)
            self.centers = kmeans.labels_
        else:
            self.centers = np.zeros([len(self.poi_info) + 1])
        print("KMeans fin.")

    def train(self):
        if not os.path.exists("./weights/"):
            os.makedirs("./weights/")
        if not os.path.exists("./weights/temp/"):
            os.makedirs("./weights/temp/")
        filepath = "./weights/temp/opnn-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        his = self.model.fit_generator(self.__generator(len(self.poi2idx)),
                                       steps_per_epoch=len(self.x_train) / self.paras.batch_size,
                                       epochs=self.paras.epoch, verbose=1,
                                       callbacks=callbacks_list)
        print(his.history['loss'][-1])

    def train_to_converge(self):
        if not os.path.exists("./weights/"):
            os.makedirs("./weights/")
        if not os.path.exists("./weights/temp/"):
            os.makedirs("./weights/temp/")
        filepath = "./weights/temp/opnn-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        last_loss = 999999999999
        while True:
            his = self.model.fit_generator(self.__generator(self.x_train, self.u_train, self.d_train, self.t_train, self.y_train, len(self.poi2idx)),
                                           steps_per_epoch=len(self.x_train) / self.paras.batch_size,
                                           epochs=1, verbose=1,
                                           callbacks=callbacks_list)
            if last_loss / his.history['loss'][-1] < 2:
                break
            else:
               last_loss = his.history['loss'][-1]

    def evaluate(self, topN=[10], info_1="", info_2="", post_fix=""):
        if not os.path.exists("./reports/"):
            os.makedirs("./reports/")
        fout = open('./reports/report-{}-{}.txt{}'.format(info_1, info_2,post_fix), 'w')
        weights = [(f, float(f.split('-')[2].split('.hdf5')[0])) for f in listdir('./weights/temp/') if isfile(join('./weights/temp/', f))]  # get least loss weight
        weights.sort(key=lambda x: x[1])
        w = './weights/temp/' + weights[0][0]
        new_w = './weights/' + weights[0][0]
        copyfile(w, new_w)
        weights = ['./weights/temp/' + each[0] for each in weights]
        #for w in weights:
            #os.remove(w)
        fout.write(new_w + '\n')
        self.model.load_weights(new_w)

        for n in topN:
            case_count = 0
            item_count = 0
            match_count = 0.0
            for i, x in enumerate(self.x_tune):
                u = self.u_tune[i]
                #t = to_categorical(self.d_tune[i].hour, 24)
                #p = np.array([[self.poi_info[ self.idx2poi[x[-1]]][1], self.poi_info[ self.idx2poi[x[-1]]][2]]])
                 
                preds = self.model.predict([np.array([x]), np.array([u]), np.array([x[-1]]), np.array([self.d_tune[i].hour + 24 * self.d_tune[i].weekday()/5])])[0]
                preds = self.get_top_k(preds, topn=n, distance=10, reverse=True, cur_poi=self.x_tune[i][-1], cur_user=self.u_tune[i][-1], user_visit=self.train_visit)
                case_count += 1
                item_count += len(set(self.y_tune[i]))
                match_count += len(set(preds) & set(self.y_tune[i]))
            print 'Match count ', match_count
            print 'Item count ', item_count
            precision = match_count / float(case_count * n)
            recall = match_count / float(item_count)
            print 'Precision@{} : {:f}'.format(n, precision)
            fout.write('Precision@{} : {:f}\n'.format(n, precision))
            print 'Recall@{} : {:f}'.format(n, recall)
            fout.write('Recall@{} : {:f}\n'.format(n, recall))
            print 'Fmeasure@{} : {:f}'.format(n, 2. * (precision * recall) / (precision + recall))
            fout.write('Fmeasure@{} : {:f}\n\n'.format(n, 2. * (precision * recall) / (precision + recall)))
            print

        retval = precision

        for n in topN:
            case_count = 0
            item_count = 0
            match_count = 0.0
            for i, x in enumerate(self.x_val):
                u = self.u_val[i]
                #t = to_categorical(self.d_val[i].hour, 24)
                #p = np.array([[self.poi_info[ self.idx2poi[x[-1]]][1], self.poi_info[ self.idx2poi[x[-1]]][2]]])
                preds = self.model.predict([np.array([x]), np.array([u]), np.array([x[-1]]), np.array([self.d_val[i].hour + 24 * self.d_val[i].weekday()/5])])[0]
                preds = self.get_top_k(preds, topn=n, distance=10, reverse=True, cur_poi=self.x_val[i][-1], cur_user=self.u_val[i][-1], user_visit=self.train_tune_visit)
                case_count += 1
                item_count += len(set(self.y_val[i]))
                match_count += len(set(preds) & set(self.y_val[i]))
            print 'Match count ', match_count
            print 'Item count ', item_count
            precision = match_count / float(case_count * n)
            recall = match_count / float(item_count)
            print 'Precision@{} : {:f}'.format(n, precision)
            fout.write('Precision@{} : {:f}\n'.format(n, precision))
            print 'Recall@{} : {:f}'.format(n, recall)
            fout.write('Recall@{} : {:f}\n'.format(n, recall))
            print 'Fmeasure@{} : {:f}'.format(n, 2. * (precision * recall) / (precision + recall))
            fout.write('Fmeasure@{} : {:f}\n\n'.format(n, 2. * (precision * recall) / (precision + recall)))
            print
        fout.close()
        return retval

    def batch_evaluate(self, topN=[10], info_1="", info_2="", post_fix=""):
        if not os.path.exists("./reports/"):
            os.makedirs("./reports/")
        fout = open('./reports/report-{}-{}.txt{}'.format(info_1, info_2,post_fix), 'w')
        weights = [(f, float(f.split('-')[2].split('.hdf5')[0])) for f in listdir('./weights/temp/') if isfile(join('./weights/temp/', f))]  # get least loss weight
        weights.sort(key=lambda x: x[1])
        w = './weights/temp/' + weights[0][0]
        new_w = './weights/' + weights[0][0]
        copyfile(w, new_w)
        weights = ['./weights/temp/' + each[0] for each in weights]
        #for w in weights:
            #os.remove(w)
        fout.write(new_w + '\n')
        self.model.load_weights(new_w)

        # start test
        for n in topN:
            case_count = 0
            item_count = 0
            match_count = 0.0
            batch = self.paras.batch_size

            temp = [x.hour for x in self.d_tune ]
            temp = np.array(temp)
            is_week = np.array([x.weekday()/5 for x in self.d_tune ]) 
            print(time.time()) 
            preds = self.model.predict_on_batch([self.x_tune, self.u_tune, self.x_tune[:,-1], temp + 0 * is_week ])
            print(time.time()) 
            for j in range(len(self.x_tune)):
                pred = preds[j]
                pred = self.get_top_k(pred, topn=n, distance=50, reverse=True, cur_poi=self.x_tune[j][-1], cur_user=self.u_tune[j][-1], user_visit=self.train_visit)
                case_count += 1
                item_count += len(set(self.y_tune[j]))
                match_count += len(set(pred) & set(self.y_tune[j]))
            
            print 'Match count ', match_count
            print 'Item count ', item_count
            precision = match_count / float(case_count * n)
            recall = match_count / float(item_count)
            print 'Precision@{} : {:f}'.format(n, precision)
            print 'Recall@{} : {:f}'.format(n, recall)
            print 'Fmeasure@{} : {:f}'.format(n, 2. * (precision * recall) / (precision + recall))

        retval = precision

        for n in topN:
            case_count = 0
            item_count = 0
            match_count = 0.0
            batch = self.paras.batch_size

            temp = [x.hour for x in self.d_val ]
            temp = np.array(temp)
            is_week = np.array([x.weekday()/5 for x in self.d_val ]) 
 
            preds = self.model.predict_on_batch([self.x_val, self.u_val, self.x_val[:,-1], temp + 0 * is_week ])
            for j in range(len(self.x_val)):
                pred = preds[j]
                pred = self.get_top_k(pred, topn=n, distance=50, reverse=True, cur_poi=self.x_val[j][-1], cur_user=self.u_val[j][-1], user_visit=self.train_tune_visit)
                case_count += 1
                item_count += len(set(self.y_val[j]))
                match_count += len(set(pred) & set(self.y_val[j]))
            
            print 'Match count ', match_count
            print 'Item count ', item_count
            precision = match_count / float(case_count * n)
            recall = match_count / float(item_count)
            print 'Precision@{} : {:f}'.format(n, precision)
            print 'Recall@{} : {:f}'.format(n, recall)
            print 'Fmeasure@{} : {:f}'.format(n, 2. * (precision * recall) / (precision + recall))

        fout.close()
        return retval

    def batch_evaluate_dev(self, topN=[10], info_1="", info_2="", post_fix="", dist_filter=10):
        if not os.path.exists("./reports/"):
            os.makedirs("./reports/")
        fout = open('./reports/report-{}-{}.txt{}'.format(info_1, info_2,post_fix), 'w')
        weights = [(f, float(f.split('-')[2].split('.hdf5')[0])) for f in listdir('./weights/temp/') if isfile(join('./weights/temp/', f))]  # get least loss weight
        weights.sort(key=lambda x: x[1])
        w = './weights/temp/' + weights[0][0]
        new_w = './weights/' + weights[0][0]
        copyfile(w, new_w)
        weights = ['./weights/temp/' + each[0] for each in weights]
        #for w in weights:
            #os.remove(w)
        fout.write(new_w + '\n')
        self.model.load_weights(new_w)

        # start test
        for n in topN:
            case_count = 0
            item_count = 0
            match_count = 0.0

            temp = [x.hour for x in self.d_tune ]
            temp = np.array(temp)
            is_week = np.array([x.weekday()/5 for x in self.d_tune ])
            Loc = [ self.centers[x] for x in  self.x_tune[:,-1] ]
            Loc = np.array(Loc)
            Uc = [ self.u_labels[x] for x in  self.u_tune[:,-1] ]
            Uc = np.array(Uc)
            
            preds = self.model.predict_on_batch([self.x_tune, self.u_tune, self.x_tune[:,-1], Uc, temp + 24 * is_week, Loc])
            #preds = self.model.predict_on_batch([self.x_tune, self.u_tune, self.x_tune[:,-1], self.u_tune[:,-1], temp + 24 * is_week, Loc])
            #preds = self.model.predict_on_batch([self.x_tune, self.u_tune, self.x_tune[:,-1], self.u_tune[:,-1], temp + 24 * is_week + 48 * Loc])
            #preds = self.model.predict_on_batch([self.x_tune, self.u_tune, self.x_tune[:,-1], self.u_tune[:,-1], temp + 24 * is_week ])
            #preds = self.model.predict_on_batch([self.x_tune, self.u_tune, self.x_tune[:,-1], temp + 24 * is_week ])
            #preds = self.model.predict_on_batch([self.x_tune, self.u_tune])

            manager = mpc.Manager()
            return_dict = manager.dict()

            jobs = []
            for i in range(0, self.paras.max_process):
                p = mpc.Process(target=self.batch_top_k, args=(i, self.paras.max_process, preds, self.x_tune, self.u_tune, n, dist_filter, True, self.train_visit, return_dict))
                jobs.append(p)
                p.start()

            for i,x in enumerate(jobs):
                x.join()

            for j in range(len(self.x_tune)):
                pred = return_dict[j]
                case_count += 1
                item_count += len(set(self.y_tune[j]))
                match_count += len(set(pred) & set(self.y_tune[j]))
            """
            for j in range(len(self.x_tune)):
                pred = preds[j]
                pred = self.get_top_k(pred, topn=n, distance=50, reverse=True, cur_poi=self.x_tune[j][-1], cur_user=self.u_tune[j][-1], user_visit=self.train_visit)
                case_count += 1
                item_count += len(set(self.y_tune[j]))
                match_count += len(set(pred) & set(self.y_tune[j]))
            """            
            #print 'Match count ', match_count
            #print 'Item count ', item_count
            precision = match_count / float(case_count * n)
            recall = match_count / float(item_count)
            print 'Precision@{} : {:f}'.format(n, precision)
            print 'Recall@{} : {:f}'.format(n, recall)
            #print 'Fmeasure@{} : {:f}'.format(n, 2. * (precision * recall) / (precision + recall))

        retval = precision

        print()
        for n in topN:
            case_count = 0
            item_count = 0
            match_count = 0.0

            temp = [x.hour for x in self.d_val ]
            temp = np.array(temp)
            is_week = np.array([x.weekday()/5 for x in self.d_val ]) 
            Loc = [ self.centers[x] for x in  self.x_val[:,-1] ]
            Loc = np.array(Loc)
            Uc = [ self.u_labels[x] for x in  self.u_val[:,-1] ]
            Uc = np.array(Uc)
            #preds1 = self.model.predict_on_batch([self.x_val[:25000], self.u_val[:25000], self.x_val[:25000,-1], self.u_val[:25000,-1], np.array((temp + 24 * is_week + 48 * Loc)[:25000]) ])
            #preds2 = self.model.predict_on_batch([self.x_val[25000:], self.u_val[25000:], self.x_val[25000:,-1], self.u_val[25000:,-1], np.array((temp + 24 * is_week + 48 * Loc)[25000:]) ])
            #preds1 = self.model.predict_on_batch([self.x_val[:25000], self.u_val[:25000], self.x_val[:25000,-1], self.u_val[:25000,-1], np.array((temp + 24 * is_week)[:25000]), Loc[:25000] ])
            #preds2 = self.model.predict_on_batch([self.x_val[25000:], self.u_val[25000:], self.x_val[25000:,-1], self.u_val[25000:,-1], np.array((temp + 24 * is_week)[25000:]), Loc[25000:] ])
            preds1 = self.model.predict_on_batch([self.x_val[:25000], self.u_val[:25000], self.x_val[:25000,-1], Uc[:25000], np.array((temp + 24 * is_week)[:25000]), Loc[:25000] ])
            preds2 = self.model.predict_on_batch([self.x_val[25000:], self.u_val[25000:], self.x_val[25000:,-1], Uc[25000:], np.array((temp + 24 * is_week)[25000:]), Loc[25000:] ])
            preds = np.concatenate((preds1, preds2), axis=0)
            #preds = self.model.predict_on_batch([self.x_val, self.u_val, self.x_val[:,-1], self.u_val[:,-1],  temp + 24 * is_week ])
            #preds = self.model.predict_on_batch([self.x_val, self.u_val, self.x_val[:,-1], temp + 24 * is_week ])
            #preds = self.model.predict_on_batch([self.x_val, self.u_val])

            manager = mpc.Manager()
            return_dict = manager.dict()

            jobs = []
            for i in range(0, self.paras.max_process):
                p = mpc.Process(target=self.batch_top_k, args=(i, self.paras.max_process, preds, self.x_val, self.u_val, n, dist_filter, True, self.train_tune_visit, return_dict))
                jobs.append(p)
                p.start()

            for i,x in enumerate(jobs):
                x.join()

            for j in range(len(self.x_val)):
                pred = return_dict[j]
                case_count += 1
                item_count += len(set(self.y_val[j]))
                match_count += len(set(pred) & set(self.y_val[j]))
 
            #print 'Match count ', match_count
            #print 'Item count ', item_count
            precision = match_count / float(case_count * n)
            recall = match_count / float(item_count)
            print 'Precision@{} : {:f}'.format(n, precision)
            print 'Recall@{} : {:f}'.format(n, recall)
            #print 'Fmeasure@{} : {:f}'.format(n, 2. * (precision * recall) / (precision + recall))
        """
        print()
        # old
        for n in topN:
            case_count = 0
            item_count = 0
            match_count = 0.0

            temp = [x.hour for x in self.d_val ]
            temp = np.array(temp)
            is_week = np.array([x.weekday()/5 for x in self.d_val ]) 
            Loc = [ self.centers[x] for x in  self.x_val[:,-1] ]
            Loc = np.array(Loc)
            preds = self.model.predict_on_batch([self.x_val, self.u_val, self.x_val[:,-1], self.u_val[:,-1],  temp + 24 * is_week + 48 * Loc])
            #preds = self.model.predict_on_batch([self.x_val, self.u_val, self.x_val[:,-1], self.u_val[:,-1],  temp + 24 * is_week ])
            #preds = self.model.predict_on_batch([self.x_val, self.u_val, self.x_val[:,-1], temp + 24 * is_week ])
            #preds = self.model.predict_on_batch([self.x_val, self.u_val])

            manager = mpc.Manager()
            return_dict = manager.dict()

            jobs = []
            for i in range(0, self.paras.max_process):
                p = mpc.Process(target=self.batch_top_k, args=(i, self.paras.max_process, preds, self.x_val, self.u_val, n, dist_filter, True, self.train_tune_visit, return_dict))
                jobs.append(p)
                p.start()

            for i,x in enumerate(jobs):
                x.join()

            for j in range(len(self.x_val)):
                pred = return_dict[j]
                case_count += 1
                item_count += len(set(self.y_val[j]))
                match_count += len(set(pred) & set(self.y_val[j]))
 
            print 'Match count ', match_count
            print 'Item count ', item_count
            precision = match_count / float(case_count * n)
            recall = match_count / float(item_count)
            print 'Precision@{} : {:f}'.format(n, precision)
            print 'Recall@{} : {:f}'.format(n, recall)
            print 'Fmeasure@{} : {:f}'.format(n, 2. * (precision * recall) / (precision + recall))
        """
        fout.close()
        return retval, precision, recall

    def batch_top_k(self, i, total, x, ps, us, topn=10, distance=10, reverse=True, user_visit=None, return_dict=None):
        for idx in range( len(x) * i//total, len(x)*(i+1)//total):
            return_dict[idx] = self.get_top_k(x[idx], topn=topn, distance=distance, reverse=reverse,
                               cur_poi=ps[idx][-1], cur_user=us[idx][-1], user_visit=user_visit)
 

    def get_top_k(self, x, topn=None, distance=10, reverse=True, cur_poi=None, cur_user=None, user_visit=None):
        """
        Return indices of the `topn` smallest elements in array `x`, in ascending order.
        If reverse is True, return the greatest elements instead, in descending order.
        """
        x = np.asarray(x)  # unify code path for when `x` is not a np array (list, tuple...)


        """
        for idx, prob in enumerate(x):
            #if self.idxipoi[idx] == 'NO_LOC':
            #    x[idx] = 0.
            #    continue
            if idx == cur_poi:
                x[idx] = 0.
	    if self.dist_mat[cur_poi][idx] >= distance:
                x[idx] *= (distance/self.dist_mat[cur_poi][idx])
            if idx in user_visit[cur_user]:
                x[idx] = 0.
        """
        # new
        x[cur_poi] = 0.
        x[ list(user_visit[cur_user]) ] = 0.
        with np.errstate(divide='ignore', invalid='ignore'):
            x *= np.where(self.dist_mat[cur_poi] > distance, distance / self.dist_mat[cur_poi], 1.0)

        if topn is None:
            topn = x.size
        if topn <= 0:
            return []
        if reverse:
            x = -x
        if topn >= x.size or not hasattr(np, 'argpartition'):
            return np.argsort(x)[:topn]
        # np >= 1.8 has a fast partial argsort, use that!
        most_extreme = np.argpartition(x, topn)[:topn]

        return most_extreme.take(np.argsort(x.take(most_extreme)))  # resort topn into order
