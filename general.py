#! /usr/bin/python
# -*- coding: utf-8 -*-
from Mylibs.Method import *
import shutil
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import gc
import sys
# general.py [embedding_type] [nb_c]

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 11/12
    set_session(tf.Session(config=config))

    file_path, file_prefix = './PoiProcessed/', 'Gowalla' # 'Foursquare'
    embedding_path = "../embedding_" +  sys.argv[1] + "/"
    #embedding_path = "../embedding_" + file_prefix + "_simu_cos_bound_0.4_iter_100" + "/"


    fout = open('report_'+ sys.argv[1] + '.txt', 'a')
    #fout.write("nb_c,dim,drop,pre,rec\n")

    test_case = 5
    nb_c_l = [int(sys.argv[2])/100.0 ]
    print(sys.argv[1], nb_c_l)
    for nb_c in nb_c_l:
        for _ in range(test_case):
            paras = get_NN_paras()
            paras.epoch = 1
            paras.nb_cluster = 1
            paras.user_cluster = 1
            paras.augment_rate = nb_c
            NN = OPNN(paras)
            NN.load_embedding(embedding_path)
            NN.load_dist_matrix( file_path + file_prefix, sys.argv[1] )
            NN.compute_center()
            NN.build_dataset( file_path + file_prefix + "_train_6.txt",
                              file_path + file_prefix + "_tune_6.txt",
                              file_path + file_prefix + "_test_6.txt" )

            NN.compute_user_embedding_v2()
            NN.build_model()
            #NN.train_to_converge()
            NN.train()
            last_pre = 0.0
            last_v1, last_v2 = 0.0, 0.0
            for i in range(50):
                print("epoch_"+str(i+1))
                pre, v1, v2 = NN.batch_evaluate_dev(info_1="add_structure", info_2="epoch"+str(i), post_fix="", dist_filter=int(sys.argv[3]))
                gc.collect()
                if pre > last_pre:
                    last_pre = pre
                    last_v1, last_v2 = v1, v2
                else:
                    break
                NN.train()

            shutil.rmtree( "./weights/")
            fout.write(str(nb_c)+","+str(last_v1)+","+str(last_v2)+"\n")
            gc.collect()
            #print(NN.model.get_weights())
            #print(NN.model.summary())
