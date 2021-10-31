import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell,DropoutWrapper
from models import GCN
from inits import *
import gc

flags = tf.app.flags
FLAGS = flags.FLAGS


class SCOREBASE(object):
    def __init__(self, feature_size, eb_dim, max_time_len,user_fnum,item_fnum,max_len,time_inter,k_hop):
        # reset graph
        tf.reset_default_graph()
        self.max_time_len=max_time_len
        self.max_len = max_len
        self.eb_dim = eb_dim
        self.time_inter = time_inter
        self.k_hop = k_hop+1
        # input placeholders
        self.feed_dict={}
        with tf.name_scope('inputs'):
            for i in range(max_time_len):
                self.feed_dict['adj_{}'.format(i)] = tf.sparse_placeholder(tf.float32)
                self.feed_dict['time_arr_{}'.format(i)] = tf.placeholder(tf.float32, [None,1])
                self.feed_dict['user_ids_{}'.format(i)] = tf.placeholder(tf.int32, [None, user_fnum])
                self.feed_dict['item_ids_{}'.format(i)] = tf.placeholder(tf.int32, [None, item_fnum])
                self.feed_dict['non_zero_{}'.format(i)] = tf.placeholder(tf.int32)

            self.feed_dict['time'] = tf.placeholder(tf.float32, [None, ], name='time')
            self.feed_dict['uid_seqs'] = tf.placeholder(tf.int32, [None, None], name='uid_seqs')
            self.feed_dict['iid_seqs'] = tf.placeholder(tf.int32, [None, None], name='iid_seqs')

            self.feed_dict['target_user'] = tf.placeholder(tf.int32, [None, user_fnum], name='target_user')
            self.feed_dict['target_item'] = tf.placeholder(tf.int32, [None, item_fnum], name='target_item')
            self.feed_dict['label'] = tf.placeholder(tf.int32, [None, ], name='label')
            self.feed_dict['length'] = tf.placeholder(tf.int32, [None, ])
            self.feed_dict['lr'] = tf.placeholder(tf.float32, [])
            self.feed_dict['neg_shape'] = tf.placeholder(tf.int32, [2, ])
            self.feed_dict['f_dropout'] = tf.placeholder_with_default(0.0, shape=())
            self.feed_dict['adj_dropout'] = tf.placeholder_with_default(0.0, shape=())
            self.feed_dict['keep_prob'] = tf.placeholder_with_default(1.0, shape=())
            self.feed_dict['reg_lambda'] = tf.placeholder(tf.float32, [], name='lambda')
            self.feed_dict['is_training'] = tf.placeholder(tf.bool, shape=())
            self.feed_dict['uid_which_slices'] = tf.placeholder(tf.int32, [None,max_time_len])
            self.feed_dict['iid_which_slices'] = tf.placeholder(tf.int32, [None,max_time_len])
            self.feed_dict['uid_seqs_len'] = tf.placeholder(tf.int32, [None, ])
            self.feed_dict['iid_seqs_len'] = tf.placeholder(tf.int32, [None, ])

        # embedding
        with tf.name_scope('embedding'):
            self.emb_mtx = tf.Variable(tf.random_normal([feature_size,eb_dim], mean=0, stddev=0.1))
            self.t_user = tf.Variable(tf.random_normal([1], mean=0, stddev=0.1))
            self.t_item = tf.Variable(tf.random_normal([1], mean=0, stddev=0.1))
            self.slice_embedding = tf.Variable(tf.random_normal([self.eb_dim], mean=0, stddev=0.1))
            self.emb_mtx_mask = tf.constant(value=1., shape=[feature_size - 1, eb_dim])
            self.emb_mtx_mask = tf.concat([tf.constant(value=0., shape=[1, eb_dim]), self.emb_mtx_mask], axis=0)
            self.emb_mtx = self.emb_mtx * self.emb_mtx_mask
            # target item and target user
            self.target_item = tf.reduce_sum(tf.nn.embedding_lookup(self.emb_mtx, self.feed_dict['target_item']),axis=1)
            self.target_user = tf.reduce_sum(tf.nn.embedding_lookup(self.emb_mtx, self.feed_dict['target_user']),axis=1)




    def build_train_step(self):
        # optimizer and training step
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.feed_dict['lr'])
        self.train_step = self.optimizer.minimize(self.loss)

    def train(self, sess, feed_dict):
        loss, _ = sess.run([self.loss,self.train_step], feed_dict=feed_dict)
        return loss

    def eval(self,sess, feed_dict):
        y_pred,loss= sess.run([self.y_pred,self.loss],feed_dict=feed_dict)
        return y_pred,loss

    def build_fc_net(self, inp):
        fc1 = tf.layers.dense(inp, 200, activation=tf.nn.relu, name='fc1')
        dp1 = tf.nn.dropout(fc1, self.feed_dict['keep_prob'], name='dp1')
        fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.relu, name='fc2')
        dp2 = tf.nn.dropout(fc2, self.feed_dict['keep_prob'], name='dp2')
        fc3 = tf.layers.dense(dp2, 1, activation=None, name='fc3')
        # output
        self.y_pred = tf.reshape(tf.nn.sigmoid(fc3), [-1, ])

    def build_logloss(self):
        # loss
        self.log_loss = tf.losses.log_loss(self.feed_dict['label'], self.y_pred)
        self.loss = self.log_loss

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from {}'.format(path))

    def build_l2norm(self):
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.feed_dict['reg_lambda'] * tf.nn.l2_loss(v)

    def multi_GCN(self):
        iid_embedding_seqs = []
        uid_embedding_seqs = []
        iid_time_embeddings_seqs = []
        uid_time_embeddings_seqs = []
        for i in range(self.max_time_len):
            with tf.variable_scope('gcn_{}'.format(i)):
                user_ids=self.feed_dict['user_ids_{}'.format(i)]
                item_ids=self.feed_dict['item_ids_{}'.format(i)]
                user_inputs=tf.reduce_sum(tf.nn.embedding_lookup(self.emb_mtx,user_ids),axis=1)
                item_inputs=tf.reduce_sum(tf.nn.embedding_lookup(self.emb_mtx,item_ids),axis=1)
                inputs=tf.concat([user_inputs,item_inputs],axis=0)
                user_num = tf.shape(user_inputs)[0]
                item_num = tf.shape(item_inputs)[0]
                output=GCN(inputs,self.feed_dict['adj_{}'.format(i)],self.feed_dict['f_dropout'],self.feed_dict['adj_dropout'],self.feed_dict['non_zero_{}'.format(i)],
                                        user_num,item_num,self.eb_dim).build()
                output = tf.nn.dropout(output, self.feed_dict['keep_prob'])

                lookup_uid = tf.nn.embedding_lookup(tf.transpose(self.feed_dict['uid_seqs']), [i])
                lookup_iid = tf.nn.embedding_lookup(tf.transpose(self.feed_dict['iid_seqs']), [i])

                uid_embedding_seqs.append(tf.nn.embedding_lookup(output, lookup_uid))
                iid_embedding_seqs.append(tf.nn.embedding_lookup(output, lookup_iid))
             
                uid_time_embeddings_seqs.append(tf.nn.embedding_lookup(self.feed_dict['time_arr_{}'.format(i)],lookup_uid))
                iid_time_embeddings_seqs.append(tf.nn.embedding_lookup(self.feed_dict['time_arr_{}'.format(i)],lookup_iid))

        uid = tf.concat(uid_embedding_seqs, 0)
        iid = tf.concat(iid_embedding_seqs, 0)
        uid_time = tf.concat(uid_time_embeddings_seqs,0)
        iid_time = tf.concat(iid_time_embeddings_seqs,0)

        uid_seqs = tf.reshape(tf.transpose(uid, [1, 0, 2]),[-1,output.get_shape().as_list()[1]])
        iid_seqs = tf.reshape(tf.transpose(iid, [1, 0, 2]),[-1,output.get_shape().as_list()[1]])
        uid_time_seqs = tf.reshape(tf.transpose(uid_time, [1, 0, 2]), [-1, 1])
        iid_time_seqs = tf.reshape(tf.transpose(iid_time, [1, 0, 2]), [-1, 1])
       

        uid_seqs = tf.reshape(tf.nn.embedding_lookup(uid_seqs,tf.reshape(self.feed_dict['uid_which_slices'],[-1])),[-1,self.max_time_len,output.get_shape().as_list()[1]])
        iid_seqs = tf.reshape(tf.nn.embedding_lookup(iid_seqs,tf.reshape(self.feed_dict['iid_which_slices'],[-1])),[-1,self.max_time_len,output.get_shape().as_list()[1]])

        uid_time_seqs = tf.reshape(tf.nn.embedding_lookup(uid_time_seqs, tf.reshape(self.feed_dict['uid_which_slices'], [-1])),[-1, self.max_time_len, 1])
        iid_time_seqs = tf.reshape(tf.nn.embedding_lookup(iid_time_seqs, tf.reshape(self.feed_dict['iid_which_slices'], [-1])),[-1, self.max_time_len, 1])


        return uid_seqs,iid_seqs,uid_time_seqs,iid_time_seqs


    def attention(self, key, value, query,length,m_len):
        mask = -1e9*tf.cast(tf.not_equal(tf.expand_dims(tf.sequence_mask(length, m_len), 2),True),dtype=tf.float32)
        # key, value: [B, T, Dk], query: [B, Dq], mask: [B, T, 1]
        _, max_len, k_dim = key.get_shape().as_list()
        query = tf.layers.dense(query, k_dim, activation=None)
        queries = tf.tile(tf.expand_dims(query, 1), [1, max_len, 1])  # [B, T, Dk]
        inp = tf.concat([queries, key, queries - key, queries * key], axis=-1)
        fc1 = tf.layers.dense(inp, 128, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 64, activation=tf.nn.relu)
        fc3 = tf.layers.dense(fc2, 1, activation=None)  # [B, T, 1]

        score = tf.nn.softmax(fc3+mask,axis=1)  # [B, T, 1]

        return score


class PP(SCOREBASE):
    def __init__(self,feature_size, eb_dim, max_time_len,user_fnum,item_fnum,max_len):
        super(PP,self).__init__(feature_size, eb_dim, max_time_len,user_fnum,item_fnum,max_len)
        num_blocks = 1
        uid_seqs, iid_seqs,uid_time_seqs,iid_time_seqs,user_slice_seqs,item_slice_seqs = self.multi_GCN()
        gcn_emb_size = 16

        user_mask = tf.expand_dims(tf.sequence_mask(self.feed_dict['user_seq_len_batch'], max_len, dtype=tf.float32),axis=-1)
        item_mask = tf.expand_dims(tf.sequence_mask(self.feed_dict['item_seq_len_batch'], max_len, dtype=tf.float32),axis=-1)
        
        uid_time = tf.expand_dims(self.feed_dict['user_time_seq_batch'][:, 1:] - self.feed_dict['user_time_seq_batch'][:, :-1],-1) * user_mask[:, 1:, :] / (3600 * 24 * 7)
        iid_time = tf.expand_dims(self.feed_dict['item_time_seq_batch'][:, 1:] - self.feed_dict['item_time_seq_batch'][:, :-1],-1) * item_mask[:, 1:, :] / (3600 * 24 * 7)

        with tf.variable_scope('rnn'):
            # user_temporal_gate = tf.exp(-tf.layers.dense(tf.expand_dims(self.feed_dict['uid_inters'],-1),1,activation=tf.nn.relu))
            user_side_rep, user_seq_final = tf.nn.dynamic_rnn(GRUCell(gcn_emb_size), inputs=user_slice_seqs,
                                                              sequence_length=self.feed_dict['user_seq_len_batch'],
                                                              dtype=tf.float32,
                                                              scope='gru_user')

            # item_temporal_gate = tf.exp(-tf.layers.dense(tf.expand_dims(self.feed_dict['iid_inters'],-1), 1, activation=tf.nn.relu))
            item_side_rep, item_seq_final = tf.nn.dynamic_rnn(GRUCell(gcn_emb_size), inputs=item_slice_seqs,
                                                              sequence_length=self.feed_dict['item_seq_len_batch'],
                                                              dtype=tf.float32,
                                                              scope='gru_item')

        tv1 = tf.layers.dense(user_side_rep[:, :-1, :], 1,
                              kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                              bias_initializer=tf.random_normal_initializer(mean=0, stddev=0.1))  # (batch_size,seq,1)
        tv2 = tv1 + self.t_user * uid_time  # (none, seq, 1)
        tv3 = (1 / self.t_user) * tf.exp(tv1)
        tv4 = (1 / self.t_user) * tf.exp(tv2)
        f_t = tf.exp(tv2 + tv3 - tv4)  # (batch_size,1)
        loss_time_user = -1 * tf.reduce_sum(tf.log(f_t + 1e-8) * user_mask[:, :-1, :]) / (
                    tf.reduce_sum(user_mask[:, :-1, :]) + tf.exp(-8.0))
        
        tv1 = tf.layers.dense(item_side_rep[:, :-1, :], 1,
                              kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                              bias_initializer=tf.random_normal_initializer(mean=0, stddev=0.1))  # (batch_size,seq,1)
        tv2 = tv1 + self.t_item * iid_time  # (none, seq, 1)
        tv3 = (1 / self.t_item) * tf.exp(tv1)
        tv4 = (1 / self.t_item) * tf.exp(tv2)
        f_t = tf.exp(tv2 + tv3 - tv4)  # (batch_size,1)
        loss_time_item = -1 * tf.reduce_sum(tf.log(f_t + 1e-8) * item_mask[:, :-1, :]) / (
                    tf.reduce_sum(item_mask[:, :-1, :]) + tf.exp(-8.0))


        inp = tf.concat([user_seq_final, item_seq_final, self.target_item, self.target_user], axis=-1)
        
        self.build_fc_net(inp)
        self.build_logloss()
        self.loss += 0.1 * loss_time_user
        self.loss += 0.1 * loss_time_item
        self.build_l2norm()
        self.build_train_step()



