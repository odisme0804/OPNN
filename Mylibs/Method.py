import numpy as np
import h5py
from keras.models import Model
from keras.layers import Dense, Embedding, Lambda, LSTM, Input, add 
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras import metrics
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from embedding import embedding
from config import *

class PEURNN:
    def __init__(self, stacked_num, num_of_hidden_layer):
        self.beta = stacked_num
        self.alpha = num_of_hidden_layer
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.checkins = None
        self.poi2idx = None
        self.idx2poi = None
        self.nb_pois = None
        self.poi_vectors = None
        self.user_vectors = None
        self.model = None
        self.x_train, self.y_train, self.x_val, self.y_val = None, None, None, None
        self.z_val, self.u_train, self.u_val = None, None, None


      def create_embedding_matrix(self):
          embedding_matrix = np.zeros((self.nb_pois + 1, self.embedding_dim))
          for word, i in self.poi2idx.items():
              try:
                  if word == 'LOCATION_NULL':
                      embedding_vector = np.array([1.0]*self.embedding_dim)
                  else:
                      embedding_vector = self.poi_vectors.wv[word]
              except KeyError:
              embedding_vector = None
          if embedding_vector is not None:
              # poi not found in embedding index will be all-zeros.
              embedding_matrix[i] = embedding_vector
                return embedding_matrix

  def init_model(self):
                if need_embedding:
        embedding()
    self.load_training_data()
    self.load_vectors()
    self.load_distance_matrix()
                self.nb_pois = len(self.poi2idx)
                embedding_matrix = self.create_embedding_matrix()
    self.build_dataset()
                self.build_model(embedding_matrix)


  def load_training_data(self):
    print 'Loading vtrain.txt...'
    set_of_checkins = open(training_file_dir + '/vtrain.txt', 'r').read().split('\n')
    checkins = []
    for c in set_of_checkins:
      checkins.extend(c.split())
    pois = list(set(checkins))
    self.checkins = set_of_checkins
    self.poi2idx = dict((p, i) for i, p in enumerate(pois))
    self.idx2poi = dict((i, p) for i, p in enumerate(pois))


  def load_vectors(self):
    self.poi_vectors = Word2Vec.load(embedding_model_dir + '/poi.model')
    self.user_vectors = Doc2Vec.load(embedding_model_dir + '/user.model')


        def load_distance_matrix(self):
            with h5py.File('{}_distance.h5'.format(dataset), 'r') as hf:
                self.distance = hf['distance'][:]


  def build_dataset(self):
    users = []
    contexts = []
    futures = []
    answers = []
    with open(training_file_dir + '/train.txt', 'r') as fin:
      for row in fin:
        data = row.strip().split('\t')
        users.append(self.user_vectors.docvecs[data[0]])
        context = [self.poi2idx[poi] for poi in data[1].split()]
        future = [self.poi2idx[poi] for poi in data[2].split()]
        contexts.append(context)
        futures.append(future[0])
        answers.append(future)
    contexts = np.array(contexts)
    users = np.array(users)
    futures = np.array(futures)
    answers = np.array(answers)
    indices = np.arange(contexts.shape[0])
    np.random.shuffle(indices)
    X = contexts[indices]
    U = users[indices]
    y = futures[indices]
    z = answers[indices]
    nb_validation_samples = int(validation_split * X.shape[0])
    self.x_train = X[:-nb_validation_samples]
    self.u_train = U[:-nb_validation_samples]
    self.y_train = y[:-nb_validation_samples]

    x_val, u_val, y_val, z_val = [], [], [], []
    with open(testing_file_path, 'r') as fin:
      for row in fin:
        try:
          _user, _, _, _context, _answer,_future = row.strip().split('\t')
                                        user = 'USER_{}'.format(_user)
                                        answer = 'LOCATION_{}'.format(_answer)
                                        context = ['LOCATION_{}'.format(each) for each in _context.split()]
                                        if len(context) < self.max_len:
                                            context = ['LOCATION_NULL']*(self.max_len-len(context)) + context
                                        else:
                                            context = context[-self.max_len:]
                                        future = ['LOCATION_{}'.format(each) for each in _future.split()]
          future.pop(future.index(answer))
                                        future.insert(0, answer)
                                        future = [self.poi2idx[poi] for poi in future]
                                        x_val.append([self.poi2idx[poi] for poi in context])
          u_val.append(self.user_vectors.docvecs[user])
          y_val.append(future[0])
          z_val.append(future)
        except KeyError as e:
          print e
    self.x_val = np.array(x_val)
    self.y_val = np.array(y_val)
    self.z_val = np.array(z_val)
    self.u_val = np.array(u_val)


  def build_model(self, embedding_matrix):
    embedding_layer = Embedding(self.nb_pois + 1,
                self.embedding_dim,
                weights=[embedding_matrix],
                input_length=self.max_len,
                trainable=False)
    poi_input = Input(shape=(self.max_len,), name='poi_input')
    poi = embedding_layer(poi_input)
    user = Input(shape=(self.max_len, self.embedding_dim), name='user_input')
    merged = add([poi, user])
    x = LSTM(self.beta, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(merged)
    for i in range(self.alpha):
      x = LSTM(self.beta, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(x)
    x = LSTM(self.beta, dropout=0.2)(x)
    output = Dense(self.nb_pois, activation='softmax')(x)
    model = Model(inputs=[poi_input, user], outputs=output)
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
            metrics=['accuracy', metrics.top_k_categorical_accuracy])
    self.model = model


  def _generator(self, poi, user, labels, V):
                batch = 500
    while True:
      for i in range(len(poi) / batch):
        X = poi[i * batch:(i + 1) * batch]
        _U = [[u] * self.max_len for u in user[i * batch:(i + 1) * batch]]
        print self.poi2idx['LOCATION_NULL'] 
                                for k, x in enumerate(X):
                                        if x == '29943':# self.poi2idx['LOCATION_NULL']:
            _U[k] = np.zeros(self.embedding_dim)
        U = np.array(_U)
        y = to_categorical(labels[i * batch:(i + 1) * batch], V)
        yield [X, U], y


  def train(self):
    filepath = "data/weights/peurnn-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    self.model.fit_generator(self._generator(self.x_train, self.u_train, self.y_train, self.nb_pois),
              steps_per_epoch=len(self.x_train) / 500,
              epochs=100,
              verbose=1,
              callbacks=callbacks_list)


  def evaluate(self):
    fout = open('report/report-{}-{}.txt'.format(self.beta, self.alpha), 'w')
    weights = [(f, float(f.split('-')[2].split('.hdf5')[0])) for f in listdir('data/weights/') if isfile(join('data/weights/', f))]
    weights.sort(key=lambda x: x[1])
    w = 'data/weights/' + weights[0][0]
    new_w = 'data/' + weights[0][0]
    copyfile(w, new_w)
    weights = ['data/weights/' + each[0] for each in weights]
    for w in weights:
      os.remove(w)
    fout.write(w + '\n')
    model.load_weights(new_w)
    for n in topN:
      case_count = 0
      item_count = 0
      match_count = 0.0
      for i, x in enumerate(self.x_val):
        # make sure location_null's user is also null
        _u = [self.u_val[i]] * self.max_len
        for k, _x in enumerate(x):
          if _x == poi2idx['LOCATION_NULL']:
            _u[k] = np.zeros(self.embedding_dim)
        preds = self.model.predict([np.array([x]), np.array([_u])])[0]
        preds = argsort(preds, topn=n, reverse=True, idx2poi=idx2poi, distance=self.distance, current=self.idx2poi[x[-1]])
        case_count += 1
        item_count += len(set(self.z_val[i]))
        match_count += len(set(preds) & set(self.z_val[i]))
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

