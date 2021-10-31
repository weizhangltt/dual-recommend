import tensorflow as tf
import os
from score import *
from dataloader import *
import utils as u
import random
import time
from sklearn.metrics import *
import math
import pickle as pkl
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)

random.seed(2020)
np.random.seed(2020)
tf.set_random_seed(2020)

TRAIN_NEG_SAMPLE_NUM = 9
TEST_NEG_SAMPLE_NUM = 99

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_type', 'PP', 'the type of model')
flags.DEFINE_string('data_set', 'baby', 'dataset to be used')
flags.DEFINE_integer('embedding_dim', 16, 'Dimension of feature embedding')
flags.DEFINE_integer('train_batch_size', 100, 'batch size of train set')
flags.DEFINE_integer('valid_batch_size', 100, 'batch size of evaluation set')

MAX_LEN = 50
k_hop = 2


# for Music
TIME_SLICE_NUM_Music = 12
START_TIME_Music = 0
FEAT_SIZE_Music = 1 + 22886 + 28091
DATA_DIR_Music = 'data/Yelp/feateng/'
GRAPH_DIR_Music = 'data/Yelp/graph/'
USER_PER_COLLECTION_Music = 500
ITEM_PER_COLLECTION_Music = 500
USER_NUM_Music = 22886
ITEM_NUM_Music = 28091
TIME_DELTA_Yelp = 30 * 24 * 3600

# for Baby
TIME_SLICE_NUM_Baby = 21
START_TIME_Baby = 0
FEAT_SIZE_Baby = 1 + 11860 + 11314
DATA_DIR_Baby = 'data/Baby/feateng/'
GRAPH_DIR_Baby = 'data/Baby/graph/'
USER_PER_COLLECTION_Baby = 500
ITEM_PER_COLLECTION_Baby = 500
USER_NUM_Baby = 11860
ITEM_NUM_Baby = 11314
TIME_DELTA_Baby = 60 * 24 * 3600


TIME_SLICE_NUM_Netflix = 8
START_TIME_Netflix = 0
FEAT_SIZE_Netflix = 1 + 314201 + 17276
DATA_DIR_Netflix = 'data/Netflix/feateng/'
GRAPH_DIR_Netflix = 'data/Netflix/graph/'
USER_PER_COLLECTION_Netflix = 500
ITEM_PER_COLLECTION_Netflix = 500
USER_NUM_Netflix = 314201
ITEM_NUM_Netflix = 17276
TIME_DELTA_Netflix = 15 * 24 * 3600


def restore(model_type, target_file_test,pred_time_test,
            feature_size, eb_dim,max_time_len, lr, reg_lambda, graph_path,
            user_feat,item_feat,user_fnum,item_fnum,time_inter):
    print('restore begin')
    if 'PP' in model_type:
        model = PP(feature_size,eb_dim,max_time_len,user_fnum,item_fnum,MAX_LEN,time_inter,k_hop)
    else:
        print('WRONG MODEL TYPE')
        exit(1)
    model_name = '{}_{}_{}_{}'.format(model_type, FLAGS.train_batch_size, lr, reg_lambda)

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, 'save_model_{}/{}/ckpt'.format(data_set, model_name))
        print('restore eval begin')
        data_loader_test = Dataloader(FLAGS.valid_batch_size, target_file_test, TEST_NEG_SAMPLE_NUM, max_time_len,
                                       graph_path,user_feat,item_feat,MAX_LEN,pred_time_test,k_hop)
        auc, ndcg_5, ndcg_10, hr_1, hr_5, hr_10, mrr, loss = eval(model,
                                                               sess,
                                                               data_loader_test,
                                                               pred_time_test,
                                                               TEST_NEG_SAMPLE_NUM,
                                                                reg_lambda,
                                                                True)
        # p = 1. / (1 + TEST_NEG_SAMPLE_NUM)
        # rig = 1 -(logloss / -(p * math.log(p) + (1 - p) * math.log(1 - p)))
        print(
            'RESTORE, LOSS TEST: %.4f  NDCG@5 TEST: %.4f  NDCG@10 TEST: %.4f  HR@1 TEST: %.4f  HR@5 TEST: %.4f  HR@10 TEST: %.4f  MRR TEST: %.4f AUC TEST: %.4f' % (
            loss, ndcg_5, ndcg_10, hr_1, hr_5, hr_10, mrr, auc))
    return model_name,ndcg_5, ndcg_10, hr_1, hr_5, hr_10, mrr, auc

def getNDCG_at_K(ranklist, target_item, k):
    for i in range(k):
        if ranklist[i] == target_item:
            return math.log(2) / math.log(i + 2)
    return 0

def getHR_at_K(ranklist, target_item, k):
    if target_item in ranklist[:k]:
        return 1
    else:
        return 0

def getMRR(ranklist, target_item):
    for i in range(len(ranklist)):
        if ranklist[i] == target_item:
            return 1. / (i+1)
    return 0

def get_ranking_quality(preds, target_iids):
    preds = np.array(preds).reshape(-1, TEST_NEG_SAMPLE_NUM + 1).tolist()
    target_iids = np.array(target_iids).reshape(-1, TEST_NEG_SAMPLE_NUM + 1).tolist()
    pos_iids = np.array(target_iids).reshape(-1, TEST_NEG_SAMPLE_NUM + 1)[:,0].flatten().tolist()
    ndcg_5_val = []
    ndcg_10_val = []
    hr_1_val = []
    hr_5_val = []
    hr_10_val = []
    mrr_val = []

    for i in range(len(preds)):
        ranklist = list(reversed(np.take(target_iids[i], np.argsort(preds[i]))))
        target_item = pos_iids[i]
        ndcg_5_val.append(getNDCG_at_K(ranklist, target_item, 5))
        ndcg_10_val.append(getNDCG_at_K(ranklist, target_item, 10))
        hr_1_val.append(getHR_at_K(ranklist, target_item, 1))
        hr_5_val.append(getHR_at_K(ranklist, target_item, 5))
        hr_10_val.append(getHR_at_K(ranklist, target_item, 10))
        mrr_val.append(getMRR(ranklist, target_item))
    return np.mean(ndcg_5_val), np.mean(ndcg_10_val), np.mean(hr_1_val), np.mean(hr_5_val), np.mean(hr_10_val), np.mean(mrr_val)

def eval(model, sess, data_loader, pred_time, neg_sample_num, reg_lambda, isTest):
    preds = []
    labels = []
    target_iids = []
    losses = []
    t = time.time()
    a_s = []
    for batch in tqdm(range(data_loader.num_of_batch)):
        uids, iids, uid_seqs, iid_seqs, uid_which_slices,iid_which_slices, uid_seqs_len, iid_seqs_len, label, inter_time = data_loader.gen_seqs(batch)
        feed_dict = u.construct_dict(model, adj_list, user_ids_list, item_ids_list, None, reg_lambda,uids, iids, uid_seqs, iid_seqs,
                                     label, [pred_time]*len(label), [-1, 1 + neg_sample_num], 0.0, 0.0, 1.0, False,
                                     uid_which_slices,iid_which_slices, uid_seqs_len, iid_seqs_len,
                                     time_list,inter_time)
        pred, loss = model.eval(sess, feed_dict)
        preds += list(pred)
        labels += label
        losses.append(loss)
        target_iids += iids
    # logloss = log_loss(labels, preds)
    auc = roc_auc_score(labels, preds)
    loss = sum(losses) / len(losses)

    ndcg_5, ndcg_10, hr_1, hr_5, hr_10, mrr = get_ranking_quality(preds, target_iids)
    print("EVAL TIME: %.4fs" % (time.time() - t))
    return auc, ndcg_5, ndcg_10, hr_1, hr_5, hr_10, mrr, loss

def train(model_type, feature_size,eb_dim,max_time_len,
        adj_list,user_ids_list,item_ids_list,lr,reg_lambda,target_file_train,
        target_file_valid,pred_time_train,pred_time_valid,graph_path,
          user_feat,item_feat,user_fnum,item_fnum,time_inter):
    if 'PP' in model_type:
        model = PP(feature_size,eb_dim,max_time_len,user_fnum,item_fnum, MAX_LEN, time_inter, k_hop)
    else:
        print('WRONG MODEL TYPE')
        exit(1)

    # gpu settings
    gpu_options = tf.GPUOptions(allow_growth=True)
    # training process
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        train_losses_step = []
        train_losses = []

        vali_ndcgs_5 = []
        vali_ndcgs_10 = []
        vali_hrs_1 = []
        vali_hrs_5 = []
        vali_hrs_10 = []
        vali_mrrs = []
        vali_losses = []
        vali_aucs = []
        step = 0

        data_loader_valid = Dataloader(FLAGS.valid_batch_size, target_file_valid, TEST_NEG_SAMPLE_NUM, max_time_len,
                                       graph_path,user_feat,item_feat,MAX_LEN,pred_time_valid,k_hop)
        vali_auc, vali_ndcg_5, vali_ndcg_10, vali_hr_1, vali_hr_5, vali_hr_10, vali_mrr, vali_loss = eval(model,
                                                                                                   sess,
                                                                                                   data_loader_valid,
                                                                                                   pred_time_valid,
                                                                                                   TEST_NEG_SAMPLE_NUM,
                                                                                                   reg_lambda,
                                                                                                   False)

        vali_ndcgs_5.append(vali_ndcg_5)
        vali_ndcgs_10.append(vali_ndcg_10)
        vali_hrs_1.append(vali_hr_1)
        vali_hrs_5.append(vali_hr_5)
        vali_hrs_10.append(vali_hr_10)
        vali_mrrs.append(vali_mrr)
        vali_losses.append(vali_loss)
        vali_aucs.append(vali_auc)

        print(
            "STEP %d  LOSS TRAIN: NULL  LOSS VALI: %.4f  NDCG@5 VALI: %.4f  NDCG@10 VALI: %.4f  HR@1 VALI: %.4f  HR@5 VALI: %.4f  HR@10 VALI: %.4f  MRR VALI: %.4f AUC VALI: %.4f" % (
            step, vali_loss, vali_ndcg_5, vali_ndcg_10, vali_hr_1, vali_hr_5, vali_hr_10, vali_mrr, vali_auc))
        early_stop = False
        data_loader_train=Dataloader(FLAGS.train_batch_size,target_file_train,TRAIN_NEG_SAMPLE_NUM,max_time_len,graph_path,user_feat,item_feat,
                                     MAX_LEN, pred_time_train,k_hop)
        eval_iter_num = data_loader_train.num_of_batch // 3
        
        for epoch in tqdm(range(10)):
            if early_stop:
                break
            for batch in range(data_loader_train.num_of_batch):
                if early_stop:
                    break
                target_uids, target_iids,uid_seqs, iid_seqs, uid_which_slices,iid_which_slices, uid_seqs_len, iid_seqs_len, label, inter_time = data_loader_train.gen_seqs(batch)
                feed_dict_train=u.construct_dict(model,adj_list,user_ids_list,item_ids_list,lr,reg_lambda,target_uids,target_iids, uid_seqs, iid_seqs,
                                                 label,[pred_time_train]*len(label),[-1,1+TRAIN_NEG_SAMPLE_NUM],0.2,0.2,0.8,True,
                                                 uid_which_slices, iid_which_slices, uid_seqs_len, iid_seqs_len, time_list, inter_time)
                train_loss=model.train(sess,feed_dict_train)
              
                step+=1
                train_losses_step.append(train_loss)
                if step % eval_iter_num == 0:
                    train_loss = sum(train_losses_step) / len(train_losses_step)
                    train_losses.append(train_loss)
                    train_losses_step = []
                    vali_auc, vali_ndcg_5, vali_ndcg_10, vali_hr_1, vali_hr_5, vali_hr_10, vali_mrr, vali_loss = eval(model,
                                                                                                                  sess,
                                                                                                                  data_loader_valid,
                                                                                                                  pred_time_valid,
                                                                                                                  TEST_NEG_SAMPLE_NUM,
                                                                                                                  reg_lambda,
                                                                                                                  False)
                    vali_ndcgs_5.append(vali_ndcg_5)
                    vali_ndcgs_10.append(vali_ndcg_10)
                    vali_hrs_1.append(vali_hr_1)
                    vali_hrs_5.append(vali_hr_5)
                    vali_hrs_10.append(vali_hr_10)
                    vali_mrrs.append(vali_mrr)
                    vali_losses.append(vali_loss)
                    vali_aucs.append(vali_auc)

                    print(
                        "STEP %d  LOSS TRAIN: %.4f  LOSS VALI: %.4f  NDCG@5 VALI: %.4f  NDCG@10 VALI: %.4f  HR@1 VALI: %.4f  HR@5 VALI: %.4f  HR@10 VALI: %.4f  MRR VALI: %.4f AUC VALI: %.4f" % (
                        step, train_loss, vali_loss, vali_ndcg_5, vali_ndcg_10, vali_hr_1, vali_hr_5, vali_hr_10,
                        vali_mrr, vali_auc))

                    if vali_mrrs[-1] > max(vali_mrrs[:-1]):
                        # save model
                        model_name = '{}_{}_{}_{}'.format(model_type, FLAGS.train_batch_size, lr, reg_lambda)
                        if not os.path.exists('save_model_{}/{}/'.format(data_set, model_name)):
                            os.makedirs('save_model_{}/{}/'.format(data_set, model_name))
                        save_path = 'save_model_{}/{}/ckpt'.format(data_set, model_name)
                        model.save(sess, save_path)

                    if len(vali_mrrs) > 3 and epoch > 0:
                        if (vali_mrrs[-1] < vali_mrrs[-2] and vali_mrrs[-2] < vali_mrrs[-3] and vali_mrrs[-3] <
                                vali_mrrs[-4]):
                            early_stop = True
                            print('=====early stop=====')
                        elif (vali_mrrs[-1] - vali_mrrs[-2]) <= 0.001 and (vali_mrrs[-2] - vali_mrrs[-3]) <= 0.001 and (vali_mrrs[-3] - vali_mrrs[-4]) <= 0.001:
                            early_stop = True
                            print('=====early stop=====')
        # generate log
        if not os.path.exists('logs_{}/'.format(data_set)):
            os.makedirs('logs_{}/'.format(data_set))
        model_name = '{}_{}_{}_{}'.format(model_type, FLAGS.train_batch_size, lr, reg_lambda)

        with open('logs_{}/{}.pkl'.format(data_set, model_name), 'wb') as f:
            dump_tuple = (
            train_losses, vali_losses, vali_ndcgs_5, vali_ndcgs_10, vali_hrs_1, vali_hrs_5, vali_hrs_10,
            vali_mrrs)
            pkl.dump(dump_tuple, f)
        with open('logs_{}/{}.result'.format(data_set, model_name), 'w') as f:
            index = np.argmax(vali_mrrs)
            f.write('Result Validation NDCG@5: {}\n'.format(vali_ndcgs_5[index]))
            f.write('Result Validation NDCG@10: {}\n'.format(vali_ndcgs_10[index]))
            f.write('Result Validation HR@1: {}\n'.format(vali_hrs_1[index]))
            f.write('Result Validation HR@5: {}\n'.format(vali_hrs_5[index]))
            f.write('Result Validation HR@10: {}\n'.format(vali_hrs_10[index]))
            f.write('Result Validation MRR: {}\n'.format(vali_mrrs[index]))
        return vali_mrrs[index]


if __name__=='__main__':
    if len(sys.argv) < 4:
        print("PLEASE INPUT [MODEL TYPE] [GPU] [DATASET]")
        sys.exit(0)
    model_type = sys.argv[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
    data_set = sys.argv[3]

    if data_set == 'yelp':
        user_feat_dict_file = None
        item_feat_dict_file = None
        user_fnum = 1
        item_fnum = 1

        target_file_train = DATA_DIR_Music + 'target_15.txt'
        target_file_validation = DATA_DIR_Music + 'target_16_sample.txt'
        target_file_test = DATA_DIR_Music + 'target_17_sample.txt'

        start_time = START_TIME_Music
        pred_time_train = 9
        pred_time_validation = 10
        pred_time_test = 11

        # model parameter
        feature_size = FEAT_SIZE_Music
        max_time_len = TIME_SLICE_NUM_Music - START_TIME_Music - 1
        graph_dir= GRAPH_DIR_Music
        user_num = USER_NUM_Music
        time_inter = TIME_DELTA_Yelp
    elif data_set == 'baby':
        user_feat_dict_file = None
        item_feat_dict_file = None
        user_fnum = 1
        item_fnum = 1

        target_file_train = DATA_DIR_Baby + 'target_5.txt'
        target_file_validation = DATA_DIR_Baby + 'target_6_sample.txt'
        target_file_test = DATA_DIR_Baby + 'target_7_sample.txt'

        start_time = START_TIME_Baby
        pred_time_train = 18
        pred_time_validation = 19
        pred_time_test = 20

        # model parameter
        feature_size = FEAT_SIZE_Baby
        max_time_len = TIME_SLICE_NUM_Baby - START_TIME_Baby - 1
        graph_dir=GRAPH_DIR_Baby
        user_num = USER_NUM_Baby
        time_inter = TIME_DELTA_Baby
    elif data_set == 'netflix':
        user_feat_dict_file = None
        item_feat_dict_file = None
        user_fnum = 1
        item_fnum = 1

        target_file_train = DATA_DIR_Netflix + 'target_17_sample.txt'
        target_file_validation = DATA_DIR_Netflix + 'target_18_sample.txt'
        target_file_test = DATA_DIR_Netflix + 'target_19_sample.txt'

        start_time = START_TIME_Netflix
        pred_time_train = 5
        pred_time_validation = 6
        pred_time_test = 7

        # model parameter
        feature_size = FEAT_SIZE_Netflix
        max_time_len = TIME_SLICE_NUM_Netflix - START_TIME_Netflix - 1
        graph_dir= GRAPH_DIR_Netflix
        user_num = USER_NUM_Netflix
        time_inter = TIME_DELTA_Netflix
    else:
        print('WRONG DATASET NAME: {}'.format(data_set))
        exit()

    ################################## training hyper params ##################################
    # TRAINING PROCESS
    lr=5e-4
    reg_lambdas = [1e-4]
    adj_list = []
    time_list = []
    weight_list = []
    user_ids_list = []
    item_ids_list = []
    size_list = []
    for i in range(max_time_len):
        adj_list.append(u.preprocess_adj(u.load_pickle(graph_dir + 'adj_{}'.format(i))))
        time_list.append(u.load_pickle(graph_dir + 'time_arr_{}'.format(i)))
        user_ids_list.append(u.load_pickle(graph_dir + 'user_ids_{}'.format(i)))
        item_ids_list.append(u.load_pickle(graph_dir + 'item_ids_{}'.format(i)))

    user_feat=None
    item_feat=None
    if user_feat_dict_file:
        user_feat=u.load_pickle(user_feat_dict_file)
    if item_feat_dict_file:
        item_feat=u.load_pickle(item_feat_dict_file)

    print(f'dataset:{data_set},model:{model_type}')
    ndcg_5_list = []
    ndcg_10_list = []
    hr_1_list = []
    hr_5_list = []
    hr_10_list = []
    mrr_list = []
    for num in range(3):
        vali_mrrs = []
        hyper_list = []
        for reg_lambda in reg_lambdas:
            vali_mrr = train(model_type, feature_size, FLAGS.embedding_dim,max_time_len, adj_list,
                  user_ids_list, item_ids_list,lr,reg_lambda,target_file_train,target_file_validation,pred_time_train,pred_time_validation,graph_dir,
                  user_feat,item_feat,user_fnum,item_fnum,time_inter)
            vali_mrrs.append(vali_mrr)
            hyper_list.append(reg_lambda)

        index = np.argmax(vali_mrrs)
        best_hyper = hyper_list[index]
        reg_lambda = 1e-4
        model_name,ndcg_5, ndcg_10, hr_1, hr_5, hr_10, mrr, auc=restore(model_type,target_file_test,pred_time_test,feature_size,FLAGS.embedding_dim,max_time_len,
                lr,reg_lambda,graph_dir,user_feat,item_feat,user_fnum,item_fnum,time_inter)

        ndcg_5_list.append(ndcg_5)
        ndcg_10_list.append(ndcg_10)
        hr_1_list.append(hr_1)
        hr_5_list.append(hr_5)
        hr_10_list.append(hr_10)
        mrr_list.append(mrr)


    with open('logs_{}/{}.test.result'.format(data_set, model_name), 'w') as f:
        f.write('Result Test NDCG@5: {}\n'.format(np.mean(ndcg_5_list)))
        f.write('Result Test NDCG@10: {}\n'.format(np.mean(ndcg_10_list)))
        f.write('Result Test HR@1: {}\n'.format(np.mean(hr_1_list)))
        f.write('Result Test HR@5: {}\n'.format(np.mean(hr_5_list)))
        f.write('Result Test HR@10: {}\n'.format(np.mean(hr_10_list)))
        f.write('Result Test MRR: {}\n'.format(np.mean(mrr_list)))

    print('Result Test NDCG@5: {}\n'.format(np.mean(ndcg_5_list)))
    print('Result Test NDCG@10: {}\n'.format(np.mean(ndcg_10_list)))
    print('Result Test HR@1: {}\n'.format(np.mean(hr_1_list)))
    print('Result Test HR@5: {}\n'.format(np.mean(hr_5_list)))
    print('Result Test HR@10: {}\n'.format(np.mean(hr_10_list)))
    print('Result Test MRR: {}\n'.format(np.mean(mrr_list)))




