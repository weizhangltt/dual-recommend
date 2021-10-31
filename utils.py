import scipy.sparse as sp
import numpy as np
import pickle

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)
    return sparse_to_tuple(adj_normalized)

def preprocess_adj1(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj+sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def load_pickle(file):
    return pickle.load(open(file, 'rb'))

def dump_pickle(file, obj):
    pickle.dump(obj, open(file, mode='wb'), protocol=4)

def construct_dict(model,adj_list,user_ids_list,item_ids_list,lr,reg_lambda,target_user,target_item,uid_seqs,iid_seqs,label,length,neg_shape,f_dropout,adj_dropout,keep_prob,is_training,
                   uid_which_slices, iid_which_slices, uid_seqs_len, iid_seqs_len, time_list,time):
    feed_dict = {}
    for i in range(len(adj_list)):
        feed_dict.update({model.feed_dict['adj_{}'.format(i)]: adj_list[i]})
        feed_dict.update({model.feed_dict['time_arr_{}'.format(i)]: time_list[i]})
        feed_dict.update({model.feed_dict['non_zero_{}'.format(i)]: adj_list[i][1].shape})
        feed_dict.update({model.feed_dict['user_ids_{}'.format(i)]: user_ids_list[i]})
        feed_dict.update({model.feed_dict['item_ids_{}'.format(i)]: item_ids_list[i]})

    feed_dict.update({model.feed_dict['time']: time})
    feed_dict.update({model.feed_dict['lr']: lr})
    feed_dict.update({model.feed_dict['target_user']: target_user})
    feed_dict.update({model.feed_dict['target_item']: target_item})
    feed_dict.update({model.feed_dict['uid_seqs']: uid_seqs})
    feed_dict.update({model.feed_dict['iid_seqs']: iid_seqs})
    feed_dict.update({model.feed_dict['label']: label})
    feed_dict.update({model.feed_dict['length']: length})
    feed_dict.update({model.feed_dict['neg_shape']: neg_shape})
    feed_dict.update({model.feed_dict['f_dropout']: f_dropout})
    feed_dict.update({model.feed_dict['adj_dropout']: adj_dropout})
    feed_dict.update({model.feed_dict['keep_prob']: keep_prob})
    feed_dict.update({model.feed_dict['reg_lambda']: reg_lambda})
    feed_dict.update({model.feed_dict['is_training']: is_training})
    feed_dict.update({model.feed_dict['uid_which_slices']: uid_which_slices})
    feed_dict.update({model.feed_dict['iid_which_slices']: iid_which_slices})
    feed_dict.update({model.feed_dict['uid_seqs_len']: uid_seqs_len})
    feed_dict.update({model.feed_dict['iid_seqs_len']: iid_seqs_len})

    return feed_dict