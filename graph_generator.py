import random
import pickle as pkl
import time
import numpy as np
import multiprocessing
import sys
sys.path.append('..')
import utils as u
import scipy.sparse as sp

WORKER_N = 5

# Music dataset parameters
DATA_DIR_Music = 'data/Yelp/feateng/'
TIME_SLICE_NUM_Music = 12
USER_NUM_Music = 22886
ITEM_NUM_Music = 28091
USER_PER_COLLECTION_Music = 500
ITEM_PER_COLLECTION_Music = 500
START_TIME_Music = 0

# Baby dataset parameters
DATA_DIR_Baby = 'data/Baby/feateng/'
TIME_SLICE_NUM_Baby = 21
USER_NUM_Baby = 11860
ITEM_NUM_Baby = 11314
USER_PER_COLLECTION_Baby = 500
ITEM_PER_COLLECTION_Baby = 500
START_TIME_Baby = 0


DATA_DIR_Netflix = 'data/Netflix/feateng/'
TIME_SLICE_NUM_Netflix = 8
USER_NUM_Netflix = 314201
ITEM_NUM_Netflix = 17276
USER_PER_COLLECTION_Netflix = 500
ITEM_PER_COLLECTION_Netflix = 500
START_TIME_Netflix = 0


class GraphHandler(object):
    def __init__(self,
                 time_slice_num,
                 path_to_store,
                 user_num,
                 item_num,
                 start_time,
                 user_per_collection,
                 item_per_collection
                 ):
        self.user_num = user_num
        self.item_num = item_num
        self.start_time = start_time
        self.time_slice_num = time_slice_num

        self.user_per_collection = user_per_collection
        self.item_per_collection = item_per_collection

        self.user_coll_num = self.user_num // self.user_per_collection
        if self.user_num % self.user_per_collection != 0:
            self.user_coll_num += 1
        self.item_coll_num = self.item_num // self.item_per_collection
        if self.item_num % self.item_per_collection != 0:
            self.item_coll_num += 1
        self.user_colls = [u.load_pickle(path_to_store + 'user_seq_%d' % i) for i in range(self.user_coll_num)]
        self.item_colls = [u.load_pickle(path_to_store + 'item_seq_%d' % i) for i in range(self.item_coll_num)]

    def build_graph(self,graph_path,user_feat,item_feat):
        for i in range(self.time_slice_num-1):

            slice_set_user = []
            slice_set_item = []
            for item_coll in self.item_colls:
                for item in item_coll:
                    if len(item['1hop'][i])>0:
                        slice_set_item.append(item['iid'])
                        slice_set_user.extend([j[0] for j in item['1hop'][i]])

            for user_coll in self.user_colls:
                for user in user_coll:
                    if len(user['1hop'][i])>0:
                        slice_set_user.append(user['uid'])
                        slice_set_item.extend([j[0] for j in user['1hop'][i]])

            uid_length=len(set(slice_set_user))
            iid_length=len(set(slice_set_item))
            remap_id=0
            rows_user=[]
            rows_item=[]
            uid_remap_dict={}
            iid_remap_dict={}
            slice_set_user=list(set(slice_set_user))
            slice_set_item=list(set(slice_set_item))
            for uid in slice_set_user:
                uid_remap_dict[uid]=remap_id
                rows_user.append(uid)
                remap_id+=1
            for iid in slice_set_item:
                iid_remap_dict[iid]=remap_id
                rows_item.append(iid)
                remap_id+=1

            row=[]
            col=[]
            weight=[]
            time_array = np.zeros((uid_length+iid_length,1))
            for user_coll in self.user_colls:
                for user in user_coll:
                    if len(user['1hop'][i])>0:
                        hop_list = user['1hop'][i]
                        for iid in hop_list:
                            col.append(iid_remap_dict[iid[0]])
                        length = len(hop_list)
                        row.extend([uid_remap_dict[user['uid']]] * length)
                        time_array[uid_remap_dict[user['uid']],0] = max([j[1] for j in hop_list])
                        weight.extend([1]*length)

            for item_coll in self.item_colls:
                for item in item_coll:
                    if len(item['1hop'][i])>0:
                        hop_list = item['1hop'][i]
                        for uid in hop_list:
                            col.append(uid_remap_dict[uid[0]])
                        length = len(hop_list)
                        row.extend([iid_remap_dict[item['iid']]] * length)
                        time_array[iid_remap_dict[item['iid']], 0] = max([j[1] for j in hop_list])
                        weight.extend([1]*length)

            with open(graph_path+'remap_dict_file_uid_{}'.format(i),'wb') as f:
                pkl.dump(uid_remap_dict, f)
            with open(graph_path+'remap_dict_file_iid_{}'.format(i),'wb') as f:
                pkl.dump(iid_remap_dict, f)
            print('remap ids completed')

            adj = sp.csr_matrix(
                (weight, (row, col)), shape=(uid_length+iid_length, uid_length+iid_length))

            u.dump_pickle(graph_path+'adj_{}'.format(i),adj)
            u.dump_pickle(graph_path+'time_arr_{}'.format(i),time_array)

            print('adj matrix completed')

            print(len(rows_user))
            print(len(rows_item))


            target_user_batch=[]
            target_item_batch=[]
            if user_feat == None:
                for uid in rows_user:
                    target_user_batch.append([uid])
            else:
                for uid in rows_user:
                    target_user_batch.append([uid] + user_feat[str(uid)])
            if item_feat == None:
                for iid in rows_item:
                    target_item_batch.append([iid])
            else:
                for iid in rows_item:
                    target_item_batch.append([iid] + item_feat[str(iid)])

            u.dump_pickle(graph_path + 'user_ids_{}'.format(i), target_user_batch)
            u.dump_pickle(graph_path + 'item_ids_{}'.format(i), target_item_batch)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("PLEASE INPUT [DATASET]")
        sys.exit(0)
    data_set=sys.argv[1]

    if data_set=='yelp':
        path_to_store = 'data/Yelp/1hop/'
        graph_path = 'data/Yelp/graph/'
        user_feat_dict_file = None
        item_feat_dict_file = None
        graph_handler = GraphHandler(TIME_SLICE_NUM_Music, path_to_store,USER_NUM_Music,
                                     ITEM_NUM_Music,
                                     START_TIME_Music, USER_PER_COLLECTION_Music, ITEM_PER_COLLECTION_Music)

    elif data_set=='baby':
        path_to_store = 'data/Baby/1hop/'
        graph_path = 'data/Baby/graph/'
        user_feat_dict_file = None
        item_feat_dict_file = None
        graph_handler = GraphHandler(TIME_SLICE_NUM_Baby, path_to_store,USER_NUM_Baby,
                                     ITEM_NUM_Baby,
                                     START_TIME_Baby, USER_PER_COLLECTION_Baby, ITEM_PER_COLLECTION_Baby)

    
    elif data_set=='netflix':
        path_to_store = 'data/Netflix/1hop/'
        graph_path = 'data/Netflix/graph/'
        user_feat_dict_file = None
        item_feat_dict_file = None
        graph_handler = GraphHandler(TIME_SLICE_NUM_Netflix, path_to_store,USER_NUM_Netflix,ITEM_NUM_Netflix,
                                     START_TIME_Netflix, USER_PER_COLLECTION_Netflix, ITEM_PER_COLLECTION_Netflix)


    user_feat = None
    item_feat = None
    if not user_feat_dict_file == None:
        user_feat = u.load_pickle(user_feat_dict_file)
    if not item_feat_dict_file == None:
        item_feat = u.load_pickle(item_feat_dict_file)

    graph_handler.build_graph(graph_path,user_feat,item_feat)
