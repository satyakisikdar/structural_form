import numpy as np
import networkx as nx
import pandas as pd
from Forms import ClusterGraph

def get_alpha_beta_pairs(n):
    samples = []
    while len(samples) < n:
        alpha_over_alpha_plus_beta_0 = np.random.choice(np.arange(0.05, 1, 0.1), 1)[0]
        alpha_over_alpha_plus_beta_1 = np.random.choice(np.arange(0.05, 1, 0.1), 1)[0]

        if alpha_over_alpha_plus_beta_0 <= alpha_over_alpha_plus_beta_1:
            alpha_plus_beta_0 = np.random.choice([1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16, 32], 1)[0]
            alpha_0 = alpha_over_alpha_plus_beta_0 * alpha_plus_beta_0
            beta_0 = alpha_plus_beta_0 - alpha_0

            alpha_plus_beta_1 = np.random.choice([1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16, 32], 1)[0]
            alpha_1 = alpha_over_alpha_plus_beta_1 * alpha_plus_beta_1
            beta_1 = alpha_plus_beta_1 - alpha_1

            samples.append(((round(alpha_0, 3), round(beta_0, 3)), (round(alpha_1, 3), round(beta_1, 3))))
    return samples


def get_log_like(theta, data_graph, cluster_labels):
    mat = nx.to_pandas_adjacency(data_graph, dtype=int)
    nodes = list(mat.index)

    ll = 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            u, v = nodes[i], nodes[j]
            zu, zv = cluster_labels[u], cluster_labels[v]
            if mat.iloc[i, j] == 1:
                l = np.log(theta[zu][zv])
            else:
                # print('not edge', (u, v))
                l = np.log(1 -theta[zu][zv])
            ll += l
    return round(ll, 3)


def construct_theta_matrix(s_df: pd.DataFrame, alpha_beta_pair, uniq_labels):
    '''
    given the S matrix (s_df), construct the theta matrix
    :param s_df:
    :return:
    '''
    theta_df = pd.DataFrame(s_df)
    for col in theta_df.columns:  # sets all entries to 0
        theta_df[col].values[:] = 0

    pair_0, pair_1 = alpha_beta_pair

    alpha_0, beta_0 = pair_0
    alpha_1, beta_1 = pair_1

    for i in range(theta_df.shape[0]):
        for j in range(theta_df.shape[1]):
            z_u, z_v = uniq_labels[i], uniq_labels[j]
            if s_df[z_u][z_v] > 0:
                theta_df[z_u][z_v] = np.random.beta(alpha_1, beta_1, 100).mean()
            else:
                theta_df[z_u][z_v] = np.random.beta(alpha_0, beta_0, 100).mean()
    
    return theta_df

np.random.seed(0)
data_g = nx.Graph()
data_g.add_edges_from([(1, 2), (1, 3), (2, 3), (4, 5), (4, 6), (3, 5)])
cluster_g = nx.Graph()

# cluster_g.add_node('b', members={1, 2, 3})
# cluster_g.add_node('c', members={4, 5, 6})
cluster_g.add_node('a', members={1, 2, 3, 4, 5, 6})

# z = {1: 'b', 2: 'b', 3: 'b', 4: 'c', 5: 'c', 6: 'c'}
z = {n: 'a' for n in range(1, 7)}

uniq_labels = sorted(set(z.values()))
num_clusters = len(uniq_labels)

m = np.zeros((num_clusters, num_clusters))
s_mat = pd.DataFrame(m, columns=uniq_labels, index=uniq_labels)

for cnode in cluster_g.nodes():
    for u in cluster_g.node[cnode]['members']:
        for v in data_g.neighbors(u):
            z_u, z_v = z[u], z[v]
            s_mat[z_u][z_v] += 1
# print(s_mat)
alpha_beta_pairs = get_alpha_beta_pairs(100)

ll = []
for alpha_beta_pair in alpha_beta_pairs:
    theta = construct_theta_matrix(s_mat, alpha_beta_pair, uniq_labels)
    ll.append(get_log_like(theta, data_g, z))

print(np.round(np.mean(ll), 3))