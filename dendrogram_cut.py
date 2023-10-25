# code source: https://github.com/jerrylin0809/pac-bayesian-dendrogram-cut/blob/main/dendrogram_cut.py
# description: https://towardsdatascience.com/automatic-dendrogram-cut-e019202e59a7

import numpy as np
import pandas as pd
import scipy.spatial
import scipy.cluster


class PACBayes:
    def __init__(self, k_max, method='average'):
        self.k_max = k_max
        self.method = method
            
    def fit(self, distance_matrix):
        '''
        Build linkage_stats
            css: cross sum of square when merging c1 and c2
            tss: total sum of square of merged cluster
        '''
        self.distance_matrix = distance_matrix
        self.n_data = distance_matrix.shape[0]
        self.linkage = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(distance_matrix), method=self.method, optimal_ordering=True)
        self.linkage_stats = [{'c1': 0, 'c2': 0, 'css': 0, 'tss': 0, 'indices': set()} for _ in range(2 * self.n_data - 1)]

        for i in range(self.n_data):
            self.linkage_stats[i]['c1'] = i
            self.linkage_stats[i]['c2'] = i
            self.linkage_stats[i]['indices'].add(i)

        for i, (c1, c2, _, _) in enumerate(self.linkage):
            c1 = int(c1)
            c2 = int(c2)
            self.linkage_stats[i + self.n_data]['c1'] = c1
            self.linkage_stats[i + self.n_data]['c2'] = c2
            self.linkage_stats[i + self.n_data]['indices'].update(self.linkage_stats[c1]['indices'])
            self.linkage_stats[i + self.n_data]['indices'].update(self.linkage_stats[c2]['indices'])

        for i in range(self.n_data, 2 * self.n_data - 1):
            c1 = self.linkage_stats[i]['c1']
            c2 = self.linkage_stats[i]['c2']
            c1_indices = np.asarray(list(self.linkage_stats[c1]['indices']))
            c2_indices = np.asarray(list(self.linkage_stats[c2]['indices']))

            sample_distances = distance_matrix[c1_indices, :][:, c2_indices]
            self.linkage_stats[i]['css'] = np.sum(sample_distances ** 2)
            self.linkage_stats[i]['tss'] = self.linkage_stats[i]['css'] + self.linkage_stats[c1]['tss'] + self.linkage_stats[c2]['tss']

        ### dynamic programming ###
        '''
        Dynamic programming
            kl_mat[i, k]: the number of clusters in the left branch of linkage_stat[$i], maximal cluster $k
            mss_mat[i, k]: the optimal mean square error achieved at linkage_stat[$i], maximal cluster $k
        '''
        self.kl_mat = np.zeros((self.n_data * 2 - 1, self.k_max + 1), dtype=int)
        self.mss_mat = np.zeros((self.n_data * 2 - 1, self.k_max + 1), dtype=float) + np.inf

        for i in range(self.n_data):
            self.mss_mat[i, 1] = 0

        for i in range(self.n_data, 2 * self.n_data - 1):
            self.mss_mat[i, 1] = self.linkage_stats[i]['tss'] / len(self.linkage_stats[i]['indices'])

        for i in range(self.n_data, 2 * self.n_data - 1):
            for k in range(2, self.k_max + 1):
                kl_min = 0
                mss_min = np.inf
                for kl in range(1, k):
                    tss = self.mss_mat[self.linkage_stats[i]['c1'], kl] + self.mss_mat[self.linkage_stats[i]['c2'], k - kl]
                    if tss < mss_min:
                        kl_min = kl
                        mss_min = tss
                self.kl_mat[i, k] = kl_min
                self.mss_mat[i, k] = mss_min
                
        return self

    def _get_cut_nodes(self, v, k):
        if k == 1:
            yield v
        else:
            yield from self._get_cut_nodes(self.linkage_stats[v]['c1'], self.kl_mat[v, k])
            yield from self._get_cut_nodes(self.linkage_stats[v]['c2'], k - self.kl_mat[v, k])

    def get_cluster_mss(self, k):
        total_mss = 0.
        for cid in self._get_cut_nodes(2 * self.n_data - 2, k):
            total_mss += self.linkage_stats[cid]['tss'] / len(self.linkage_stats[cid]['indices'])
        
        return total_mss

    def get_cluster_label(self, k):
        ### get flat clusters ###
        z = np.zeros(self.n_data, dtype=int) - 1
        for c, cid in enumerate(self._get_cut_nodes(2 * self.n_data - 2, k)):
            for i in self.linkage_stats[cid]['indices']:
                if z[i] != -1:
                    print(i)
                z[i] = c

        return z

    def _dirichlet_process_kl(self, n_list, alpha_):
        out = 0.
        count = alpha_
        for n in n_list:
            out -= np.log(alpha_ / count)
            count += 1
            for i in range(1, n):
                out -= np.log(i / count)
                count += 1
        return out

    def pac_bayesian_cut(self, alpha_=1., lambda_=1.):
        min_loss = np.inf
        min_loss_k = None
        loss_list = []

        for k in range(1, self.k_max + 1):
            n_list = [len(self.linkage_stats[c]['indices']) for c in self._get_cut_nodes(2 * self.n_data - 2, k)]
            total_mss = self.get_cluster_mss(k)
            loss = total_mss + self._dirichlet_process_kl(n_list, alpha_) / lambda_
            loss_list.append(loss)
            if loss < min_loss:
                min_loss = loss
                min_loss_k = k

        df = pd.DataFrame(list(zip([*range(1, self.k_max + 1)], loss_list)), columns=['k', 'loss'])
        return df