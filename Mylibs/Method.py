import numpy as np
from collections import defaultdict
import datetime
import h5py
import os
from os import listdir
from os.path import isfile, join
from shutil import copyfile
from keras.models import Model
from keras.layers import Dense, Embedding, Lambda, LSTM, Input, add, concatenate, Flatten, MaxoutDense, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras import metrics
from Mylibs.Tool import *

def get_NN_paras():
    class paras():
        pass
    paras.seq_len = 3
    paras.learn_rate = 0.05
    paras.max_iter = 10
    paras.epsilon = 0.0001
    paras.batch_size = 500
    paras.epoch = 20
    paras.hidden_layer = 3
    paras.hidden_dim = 300
    paras.train_poi = False
    paras.train_user = False

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
        self.x_train, self.u_train, self.y_train, self.d_train = None, None, None, None
        self.x_val, self.u_val, self.y_val, self.d_val = None, None, None, None
        self.x_tune, self.u_tune, self.y_tune, self.d_tune = None, None, None, None

    def load_embedding(self, path):
        # load pre-train embedding vectors
        word_dict = np.load(path + 'word_vector.npy').item()
        user_dict = np.load(path + 'user_vector.npy').item()
        # add no checkin vector
        word_dict["NO_LOC"] = {'vector' : np.zeros([1, self.paras.hidden_dim])}
        user_dict["NO_USR"] = np.zeros([1, self.paras.hidden_dim])

        self.poi2idx = dict((p, i) for i, p in enumerate(word_dict))
        self.idx2poi = dict((i, p) for i, p in enumerate(word_dict))

        self.user2idx = dict((u, i) for i, u in enumerate(user_dict))
        self.idx2user = dict((i, u) for i, u in enumerate(user_dict))

        # create words' and users' vector maxtrix
        self.poi_mat  = np.zeros([ len(word_dict), self.paras.hidden_dim ]) # NO_LOC is the largest idx
        self.user_mat = np.zeros([ len(user_dict), self.paras.hidden_dim ])
        for k,v in word_dict.items():
            self.poi_mat[ self.poi2idx[k] ] = v['vector']

        for k,v in user_dict.items():
            self.user_mat[ self.user2idx[k] ] = v

        print(self.poi_mat.shape)
        print(self.user_mat.shape)


    def build_model(self):
        poi_ebd_layer = Embedding(self.poi_mat.shape[0], self.paras.hidden_dim,
                                  weights=[self.poi_mat],
                                  input_length=self.paras.seq_len,
                                  trainable=self.paras.train_poi)
        user_ebd_layer = Embedding(self.user_mat.shape[0], self.paras.hidden_dim,
                                   weights=[self.user_mat],
                                   input_length=self.paras.seq_len,
                                   trainable=self.paras.train_user)

        poi_input = Input(shape=(self.paras.seq_len,), name='poi_input')
        poi = poi_ebd_layer(poi_input)
        user_input = Input(shape=(self.paras.seq_len,), name='user_input')
        user = user_ebd_layer(user_input)

        # how to merge 2 vec
        merged = concatenate([poi, user])

        x = LSTM(self.paras.hidden_dim, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(merged)

        #x = TimeDistributed(MaxoutDense(self.paras.hidden_dim, nb_feature=2))(x)

        #x = LSTM(self.paras.hidden_dim, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(x)

        # need if return sequnce is True
        #x = Flatten()(x)

        # add maxout
        #x = MaxoutDense(self.paras.hidden_dim, nb_feature=2)(x)

        output = Dense(self.poi_mat.shape[0], activation='softmax')(x)
        model = Model(inputs=[poi_input, user_input], outputs=output)
        adam = optimizers.Adam()
        model.compile(loss='categorical_crossentropy', optimizer=adam,
                      metrics=['accuracy', metrics.top_k_categorical_accuracy])
        self.model = model

    def build_dataset(self, train_file, tune_file, test_file):
        users = []
        contexts = []
        answers = []
        dates = []
        with open(train_file, 'r') as fin:
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
                answers.append(future[0])
                checkin_time = datetime.datetime.strptime(datestr,"%Y-%m-%d %H:%M:%S")
                dates.append(checkin_time)
                # update user_visited
                self.train_visit[self.user2idx[u]] |= set(context + future)
                self.train_tune_visit[self.user2idx[u]] |= set(context + future)

        contexts = np.array(contexts)
        users = np.array(users)
        dates = np.array(dates)
        answers = np.array(answers)

        # shuffle
        indices = np.arange(contexts.shape[0])
        np.random.shuffle(indices)
        self.x_train = contexts[indices]
        self.u_train = users[indices]
        self.d_train = dates[indices]
        self.y_train = answers[indices]

        # for check
        for i in range(self.paras.seq_len):
            print(self.idx2user[users[5][i]], self.idx2poi[contexts[5][i]], dates[0], self.idx2poi[answers[5]])

        # load tuning
        users = []
        contexts = []
        answers = []
        dates = []
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
                # update user_visited
                self.train_tune_visit[self.user2idx[u]] |= set(context + future)


        self.x_tune = np.array(contexts)
        self.u_tune = np.array(users)
        self.d_tune = np.array(dates)
        self.y_tune = np.array(answers)

        # load testing
        users = []
        contexts = []
        answers = []
        dates = []
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

        self.x_val = np.array(contexts)
        self.u_val = np.array(users)
        self.d_val = np.array(dates)
        self.y_val = np.array(answers)


    def __generator(self, poi, user, labels, V):
        batch = self.paras.batch_size
        while True:
            for i in range(len(poi) / batch):
                X = poi[i * batch:(i + 1) * batch]
                U = user[i * batch:(i + 1) * batch]
                y = to_categorical(labels[i * batch:(i + 1) * batch], V)
                yield [X, U], y

    def load_dist_matrix(self, fp):
        self.dist_mat = get_dist_matrix(fp, self.poi2idx)
        print(self.dist_mat[1 ])

    def train(self):
        if not os.path.exists("./weights/"):
            os.makedirs("./weights/")
        if not os.path.exists("./weights/temp/"):
            os.makedirs("./weights/temp/")
        filepath = "./weights/temp/opnn-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        self.model.fit_generator(self.__generator(self.x_train, self.u_train, self.y_train, len(self.poi2idx)),
                                 steps_per_epoch=len(self.x_train) / self.paras.batch_size,
                                 epochs=self.paras.epoch, verbose=1,
                                 callbacks=callbacks_list)

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
                preds = self.model.predict([np.array([x]), np.array([u])])[0]
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
                preds = self.model.predict([np.array([x]), np.array([u])])[0]
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


    def get_top_k(self, x, topn=None, distance=10, reverse=True, cur_poi=None, cur_user=None, user_visit=None):
        """
        Return indices of the `topn` smallest elements in array `x`, in ascending order.
        If reverse is True, return the greatest elements instead, in descending order.
        """
        x = np.asarray(x)  # unify code path for when `x` is not a np array (list, tuple...)
        for idx, prob in enumerate(x):
            #if self.idxipoi[idx] == 'NO_LOC':
            #    x[idx] = 0.
            #    continue
            if idx == cur_poi:
                x[idx] = 0.
	    if self.dist_mat[cur_poi][idx] >= distance:
                x[idx] *= (10/self.dist_mat[cur_poi][idx])
            if idx in user_visit[cur_user]:
                x[idx] = 0.

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
