#! /usr/bin/python
# -*- coding: utf-8 -*-
from Mylibs.Method import *
import shutil

if __name__ == '__main__':
    embedding_path = "../embedding_huff/"
    file_path, file_prefix = './PoiProcessed/', 'Gowalla'


    fout = open('report_huff.txt', 'w') 
    fout.write("dim,drop,pre,rec\n")


    plist = [(300,0.2),(300,0.2),(300,0.2),(300,0.2),(300,0.2),
             (300,0.2),(300,0.2),(300,0.2),(300,0.2),(300,0.2)]

    for dim, drop in plist:
        paras = get_NN_paras()
        paras.epoch = 2
        paras.hidden_dim = dim
        paras.drop_rate = drop
        NN = OPNN(paras)
        NN.load_embedding(embedding_path)
        NN.build_dataset( file_path + file_prefix + "_train_6.txt",
                          file_path + file_prefix + "_tune_6.txt",
                          file_path + file_prefix + "_test_6.txt" )
        NN.load_dist_matrix( file_path + file_prefix)
        NN.build_model()
        last_pre = 0.0
        last_v1, last_v2 = 0.0, 0.0
        for i in range(2,200,2):
            NN.train()
            pre, v1, v2 = NN.batch_evaluate_dev(info_1="add_structure", info_2="epoch"+str(i), post_fix="")
            if pre > last_pre:
                last_pre = pre
                last_v1, last_v2 = v1, v2
            else:
                break
        shutil.rmtree( "./weights/")
        fout.write(str(dim)+","+str(drop)+","+str(last_v1)+","+str(last_v2)+"\n")
