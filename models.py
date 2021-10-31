from layers import *
from tensorflow.python.ops.rnn_cell import GRUCell

flags = tf.app.flags
FLAGS = flags.FLAGS

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        self.layers = []
        self.activations = []
        self.inputs = None

    def build(self):
        raise NotImplementedError

class GCN(Model):
    def __init__(self, inputs, adj, f_dropout, adj_dropout, num_support_nonzero,user_num,item_num,eb_dim,**kwargs):
        super(GCN, self).__init__(**kwargs)
        self.inputs = inputs
        self.input_dim = FLAGS.embedding_dim
        self.adj = adj
        self.f_dropout = f_dropout
        self.adj_dropout = adj_dropout
        self.num_support_nonzero = num_support_nonzero
        self.user_num = user_num
        self.item_num = item_num
        self.eb_dim = eb_dim

    def build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.input_dim,
                                            support=self.adj,
                                            f_dropout=self.f_dropout,
                                            adj_dropout=self.adj_dropout,
                                            num_support_nonzero=self.num_support_nonzero,
                                            user_num=self.user_num,
                                            item_num=self.item_num
                                            ))

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.input_dim,
                                            support=self.adj,
                                            f_dropout=self.f_dropout,
                                            adj_dropout=self.adj_dropout,
                                            num_support_nonzero=self.num_support_nonzero,
                                            user_num=self.user_num,
                                            item_num=self.item_num
                                            ))

        # self.layers.append(GraphConvolution(input_dim=self.input_dim,
        #                                     output_dim=self.input_dim,
        #                                     support=self.adj,
        #                                     f_dropout=self.f_dropout,
        #                                     adj_dropout=self.adj_dropout,
        #                                     num_support_nonzero=self.num_support_nonzero,
        #                                     user_num=self.user_num,
        #                                     item_num=self.item_num
        #                                     ))
        #
        # self.layers.append(GraphConvolution(input_dim=self.input_dim,
        #                                     output_dim=self.input_dim,
        #                                     support=self.adj,
        #                                     f_dropout=self.f_dropout,
        #                                     adj_dropout=self.adj_dropout,
        #                                     num_support_nonzero=self.num_support_nonzero,
        #                                     user_num=self.user_num,
        #                                     item_num=self.item_num
        #                                     ))
        #
        # self.layers.append(GraphConvolution(input_dim=self.input_dim,
        #                                     output_dim=self.input_dim,
        #                                     support=self.adj,
        #                                     f_dropout=self.f_dropout,
        #                                     adj_dropout=self.adj_dropout,
        #                                     num_support_nonzero=self.num_support_nonzero,
        #                                     user_num=self.user_num,
        #                                     item_num=self.item_num
        #                                     ))


        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)

        for i in range(len(self.activations)):
            self.activations[i]=tf.expand_dims(self.activations[i],1)
        outputs = tf.concat(self.activations,axis=1)
        user_output,item_output = tf.split(outputs,[self.user_num,self.item_num],0)
        user_rep_t, user_outputs = tf.nn.dynamic_rnn(GRUCell(self.eb_dim), inputs=user_output,dtype=tf.float32,scope='u')
        item_rep_t, item_outputs = tf.nn.dynamic_rnn(GRUCell(self.eb_dim), inputs=item_output, dtype=tf.float32,scope='i')

        return tf.concat([user_outputs,item_outputs],axis=0)




