#! /usr/bin/python
# -*- coding: utf-8 -*-
from Mylibs.Method import *

if __name__ == '__main__':
    embedding_path = "../embedding_ws1/"
    file_path, file_prefix = './PoiProcessed/', 'Gowalla'

    paras = get_NN_paras()
    paras.epoch = 2
    NN = OPNN(paras)
    NN.load_embedding(embedding_path)
    NN.build_dataset( file_path + file_prefix + "_train_6.txt",
                      file_path + file_prefix + "_tune_6.txt",
                      file_path + file_prefix + "_test_6.txt" )
    NN.load_dist_matrix( file_path + file_prefix)
    NN.build_model()
    last_pre = 0.0
    for i in range(2,200,2):
        NN.train()
        pre = NN.evaluate(info_1="update_visited", info_2="epoch"+str(i), post_fix="")
        if pre > last_pre:
            last_pre = pre
        else:
            break
