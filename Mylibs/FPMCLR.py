import random
import heapq
import timeit
import numpy as np
from Models.Tools import *


class FPMCLR:
    def __init__(self, user_set, item_set, n_factor, gh_dict, gh_tree, geo_dict, learn_rate, regular):
        self.user_set = user_set
        self.item_set = item_set

        self.gh_dict = gh_dict
        self.gh_tree = gh_tree
        self.geo_dict = geo_dict

        self.n_user = len(user_set)
        self.n_item = len(item_set)

        self.n_factor = n_factor
        self.learn_rate = learn_rate
        self.regular = regular

    def init_model(self, std=0.01):
        self.VUL = np.random.normal(0, std, size=(self.n_user, self.n_factor))
        self.VLU = np.random.normal(0, std, size=(self.n_item, self.n_factor))
        self.VLI = np.random.normal(0, std, size=(self.n_item, self.n_factor))
        self.VIL = np.random.normal(0, std, size=(self.n_item, self.n_factor))
        self.VUL_m_VLU = np.dot(self.VUL, self.VLU.T)
        self.VLI_m_VIL = np.dot(self.VLI, self.VIL.T)

    def compute_x(self, u, l, l_tm1):
        acc_val = 0.0
        for i in l_tm1:
            acc_val += np.dot(self.VLI[l, :], self.VIL[i, :])
        return np.dot(self.VUL[u, :], self.VLU[l, :]) + (acc_val / len(l_tm1))

    def compute_x_batch(self, u, l_tm1):
        former = self.VUL_m_VLU[u]
        latter = np.mean(self.VLI_m_VIL[:, l_tm1], axis=1).T
        return former + latter

    def evaluation(self, data_list, visited=None):
        self.VUL_m_VLU, self.VLI_m_VIL = [], []
        self.VUL_m_VLU = np.dot(self.VUL, self.VLU.T)
        self.VLI_m_VIL = np.dot(self.VLI, self.VIL.T)
        prefix_dict = prefix_tree(self.gh_tree)

        case_count = 0.0
        item_count = 0.0
        cc_5, cc_10, cc_20 = 0.0, 0.0, 0.0
        for (u, ans, ans_poi, l_tm1, cur_poi) in data_list:
            ans.add(ans_poi)
            cur_grid = self.gh_dict[cur_poi]
            loc_poi = self.geo_dict[cur_poi]
            grid_area = candidate_grid(loc_poi, cur_grid, prefix_dict, dist=1)
            grid_pois = poi_in_grid(grid_area, self.gh_tree)
            candidate_pois = list(set(grid_pois) - set(visited[u]))
            scores = self.compute_x_batch(u, l_tm1)
            top_5 = heapq.nlargest(5, candidate_pois, scores.take)
            top_10 = heapq.nlargest(10, candidate_pois, scores.take)
            top_20 = heapq.nlargest(20, candidate_pois, scores.take)

            if type(ans) is int:
                if ans in top_5:
                    cc_5 += 1
                if ans in top_10:
                    cc_10 += 1
                if ans in top_20:
                    cc_20 += 1
                item_count += 1
            else:
                cc_5 += len(set(ans) & set(top_5))
                cc_10 += len(set(ans) & set(top_10))
                cc_20 += len(set(ans) & set(top_20))
                item_count += len(ans)
            case_count += 1
        try:
            pre_5 = cc_5 / float(5 * case_count)
            rec_5 = cc_5 / float(item_count)
            pre_10 = cc_10 / float(10 * case_count)
            rec_10 = cc_10 / float(item_count)
            pre_20 = cc_20 / float(20 * case_count)
            rec_20 = cc_20 / float(item_count)
        except ZeroDivisionError:
            pre_5, rec_5 = 0.0, 0.0
            pre_10, rec_10 = 0.0, 0.0
            pre_20, rec_20 = 0.0, 0.0
        print 'Precision @ ' + str(5) + ' : ' + str(pre_5)
        print 'Recall @ ' + str(5) + ' : ' + str(rec_5)
        print 'Precision @ ' + str(10) + ' : ' + str(pre_10)
        print 'Recall @ ' + str(10) + ' : ' + str(rec_10)
        print 'Precision @ ' + str(20) + ' : ' + str(pre_20)
        print 'Recall @ ' + str(20) + ' : ' + str(rec_20)

    def learn_epoch(self, tr_data, neg_batch_size):
        for iter_idx in range(len(tr_data)):
            (u, i, l_t, l_tm1) = random.choice(tr_data)
            exclu_set = set()
            for l in l_tm1:
                exclu_set |= set(neighbor_poi(self.gh_dict[l], self.gh_tree))
            exclu_set = exclu_set - set(l_t)

            try:
                j_list = random.sample(exclu_set, neg_batch_size)
            except ValueError:
                j_list = exclu_set

            z1 = self.compute_x(u, i, l_tm1)
            for j in j_list:
                z2 = self.compute_x(u, j, l_tm1)
                delta = 1 - sigmoid(z1 - z2)

                VUL_update = self.learn_rate * (delta * (self.VLU[i, :] - self.VLU[j, :]) - self.regular * self.VUL[u])
                VLUi_update = self.learn_rate * (delta * self.VUL[u, :] - self.regular * self.VLU[i, :])
                VLUj_update = self.learn_rate * (-delta * self.VUL[u, :] - self.regular * self.VLU[j, :])

                self.VUL[u] += VUL_update
                self.VLU[i, :] += VLUi_update
                self.VLU[j, :] += VLUj_update

                eta = np.mean(self.VIL[l_tm1], axis=0)
                VLIi_update = self.learn_rate * (delta * eta - self.regular * self.VLI[i])
                VLIj_update = self.learn_rate * (-delta * eta - self.regular * self.VLI[j])
                VIL_update = self.learn_rate * ((delta * (self.VLI[i] - self.VLI[j]) /
                                                 len(l_tm1)) - self.regular * self.VIL[l_tm1])

                self.VLI[i] += VLIi_update
                self.VLI[j] += VLIj_update
                self.VIL[l_tm1] += VIL_update

    def learnSBPR_FPMCLR(self, tr_data, te_data,
                         visited=None, n_epoch=100, neg_batch_size=1):
        learn_timer_start = timeit.default_timer()
        for epoch in range(n_epoch):
            self.learn_epoch(tr_data, neg_batch_size=neg_batch_size)
            print ('epoch %d done' % epoch)
        learn_timer_end = timeit.default_timer()
        learn_time = learn_timer_end - learn_timer_start

        print 'FPMCLR'
        query_timer_start = timeit.default_timer()
        self.evaluation(anslr_dicttolist(te_data), visited)
        query_timer_end = timeit.default_timer()
        query_time = query_timer_end - query_timer_start
        print 'Learn time : ' + str(learn_time / float(3600)) + ' hours'
        print 'Query time : ' + str(query_time / float(3600)) + ' hours'
        print 'Total time : ' + str((learn_time + query_time) / float(3600)) + ' hours'

