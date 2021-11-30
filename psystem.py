# 
# Please cite one of the following papers if you use this code:
#
# Tarczali, T., Lehotay-Kéry, P., & Kiss, A. (2020, September). 
# Membrane Clustering Using the PostgreSQL Database 
# Management System. In Proceedings of SAI Intelligent Systems 
# Conference (pp. 377-388). Springer, Cham.
# 
# and
# 
# Lehotay-Kéry, P., Tarczali, T., & Kiss, A. (2021). P System–Based 
# Clustering Methods Using NoSQL Databases. Computation, 
# 9(10), 102.
#

import random as rd
import numpy as np
import scipy.spatial as spat
from sklearn import datasets

def load_data(l):
    return np.array(l.data,dtype=float)

def dists_squared(x: np.ndarray, y: np.ndarray):
    return spat.distance.cdist(x, y) ** 2

def part_matr(dists_sqd: np.ndarray):
    dists_nonzero = np.fmax(dists_sqd, np.finfo(np.float64).eps)
    dists_inv_sqrt = np.sqrt(np.reciprocal(dists_nonzero))
    part_matr = dists_inv_sqrt.T / np.sum(dists_inv_sqrt, axis=1)
    return part_matr

def fcm_ind(data: np.ndarray, centr: np.ndarray):
    dists_sqd = dists_squared(data, centr)
    p_matr = part_matr(dists_sqd)
    FCM = np.sum((p_matr ** 2).T * dists_sqd)
    return FCM

def memb_clust(X, n_step, n_cell, n_obj, n_clust, n_dim, c_1, c_2, c_3, w_max, w_min):

    cells = np.ndarray(shape=(n_cell, n_obj, n_clust, n_dim), dtype=float)

    best_pos = np.ndarray(shape=(n_cell, n_obj, n_clust, n_dim), dtype=float)
    best_pos_val = np.ndarray(shape=(n_cell, n_obj), dtype=float)

    e_bests = np.ndarray(shape=(n_cell, n_clust, n_dim), dtype=float)
    e_best_vals = np.ndarray(shape=(n_cell), dtype=float)

    l_bests = np.ndarray(shape=(n_cell, n_clust, n_dim), dtype=float)
    l_best_vals = np.ndarray(shape=(n_cell), dtype=float)

    g_best = np.ndarray(shape=(n_clust, n_dim), dtype=float)
    g_best_val = np.inf

    for c in range(n_cell):
        obj_vals = []
        for o in range(n_obj):
            ind = np.random.choice(list(range(len(X))), n_clust, replace=False)
            cells[c, o] = X[ind].copy()

            validity = fcm_ind(X, cells[c, o])
            obj_vals.append(validity)

            best_pos[c, o] = np.copy(cells[c, o])
            best_pos_val[c, o] = obj_vals[o]

        e_best_ind = np.argmin(obj_vals)
        e_best_vals[c] = obj_vals[e_best_ind]
        e_bests[c] = np.copy(cells[c, e_best_ind])

        l_best_vals[c] = e_best_vals[c]
        l_bests[c] = np.copy(e_bests[c])

        if l_best_vals[c] <= g_best_val:
            g_best_val = l_best_vals[c]
            g_best = np.copy(l_bests[c])

    step = 0

    vals = np.ndarray(shape=(n_cell, n_obj), dtype=float)

    while step < n_step:
        w = (w_max - (w_max - w_min) * step / n_step)
        for c in range(n_cell):
            r = list(range(n_cell))
            r.remove(c)
            eBestI = rd.choice(r)
            for o in range(n_obj):
                mul1 = c_1 * rd.uniform(0, 1)
                mul2 = c_2 * rd.uniform(0, 1)
                mul3 = c_3 * rd.uniform(0, 1)

                mul1 = c_1 * 1
                mul2 = c_2 * 1
                mul3 = c_3 * 1

                comp1 = mul1*(best_pos[c, o]-cells[c, o])
                comp2 = mul2*(e_bests[eBestI]-cells[c, o])
                comp3 = mul3*(l_bests[c]-cells[c, o])

                velocity = w*cells[c, o] + comp1 + comp2 + comp3
                cells[c, o] = cells[c, o] + velocity

                validity = fcm_ind(X, cells[c, o])
                vals[c, o] = validity

                if vals[c, o] <= best_pos_val[c, o]:
                    best_pos[c, o] = np.copy(cells[c, o])
                    best_pos_val[c, o] = vals[c, o]

        for c in range(n_cell):
            e_best_ind = np.argmin(vals[c])
            e_best_vals[c] = vals[c, e_best_ind]
            e_bests[c] = np.copy(cells[c, e_best_ind])

            if e_best_vals[c] <= l_best_vals[c]:
                l_best_vals[c] = e_best_vals[c]
                l_bests[c] = np.copy(e_bests[c])

            if e_best_vals[c] <= g_best_val:
                g_best_val = e_best_vals[c]
                g_best = np.copy(e_bests[c])
        step += 1
    return g_best

# ---------------------------------

# Input parameters:
n_step = 50
n_cell = 4
n_obj = 40
w_max = 0.9
w_min = 0.2
c_1 = 1.0
c_2 = 0.1
c_3 = 1.8

# Load dataset:
n_clust = 3
X = load_data(datasets.load_iris()) 
n_dim = X.shape[1]

# Clustering:
res = memb_clust(X, n_step, n_cell, n_obj, n_clust, n_dim, c_1, c_2, c_3, w_max, w_min)

# Create readable results:
dists = dists_squared(X, res)
res_p_matr = part_matr(dists)
res_clusters = np.argmax(res_p_matr.T, axis=1)

count = 1
d = {}
for label in res_clusters:
    if label not in d:
        d[label] = count
        count += 1

# Cluster label for each data point:
final_result = [d[i] for i in res_clusters]

print(final_result)