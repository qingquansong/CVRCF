import tensorflow as tf

STD = 1e-2


# Function to convert batch input data to use scan ops of tensorflow.
def process_batch_input_for_rnn(batch_input):
    """
    Process tensor of size [5,3,2] to [3,5,2]
    """
    batch_input_ = tf.transpose(batch_input, perm=[2, 0, 1])
    x = tf.transpose(batch_input_)

    return x


class CVRCF(object):
    """
    Coupled Variational Recurrent Collaborative Filtering
    """

    def __init__(self, args):

        self.interval_u = 86400 * args.interval_u
        self.interval_v = 86400 * args.interval_v

        # Initialization of given values
        self.num_of_u, self.num_of_v = args.num_of_u, args.num_of_v
        self.gru_input_siz_u, self.gru_input_siz_v = args.gru_input_siz_u, args.gru_input_siz_v
        self.siz_u_st, self.siz_v_st = args.siz_u_st, args.siz_v_st
        self.siz_u_mu, self.siz_v_mu = args.siz_u_dy, args.siz_v_dy
        self.siz_u_sig, self.siz_v_sig = args.siz_u_dy, args.siz_v_dy
        self.hidden_state_siz_u, self.hidden_state_siz_v = args.hidden_state_siz_u, args.hidden_state_siz_v

        self.siz_u = self.siz_u_mu + self.siz_u_sig
        self.siz_v = self.siz_v_mu + self.siz_v_sig
        self.siz_h_u = self.siz_u
        self.siz_h_v = self.siz_v

        # Set rating interval, e.g., for Movielens-10M, it should be [0, 5]
        self.max_rating = args.rating_upper_bound
        self.min_rating = args.rating_lower_bound

        ############################## Define Trainable Weights #############################
        # Stationary latent factors
        self.U_st = tf.Variable(
            tf.truncated_normal([self.num_of_u, self.siz_u_st], mean=0, stddev=STD))
        self.V_st = tf.Variable(
            tf.truncated_normal([self.num_of_v, self.siz_v_st], mean=0, stddev=STD))

        # Weights for embedding raw input to the input of GRU_u/v
        self.W_embd_u_1 = tf.Variable(
            tf.truncated_normal([self.num_of_v + 3, self.gru_input_siz_u], mean=0, stddev=STD))
        self.W_embd_v_1 = tf.Variable(
            tf.truncated_normal([self.num_of_u + 3, self.gru_input_siz_v], mean=0, stddev=STD))

        # Weights for GRU_u
        self.Wx_u = tf.Variable(
            tf.truncated_normal([self.gru_input_siz_u + 2, self.hidden_state_siz_u], mean=0, stddev=STD))
        self.Wr_u = tf.Variable(
            tf.truncated_normal([self.gru_input_siz_u + 2, self.hidden_state_siz_u], mean=0, stddev=STD))
        self.Wz_u = tf.Variable(
            tf.truncated_normal([self.gru_input_siz_u + 2, self.hidden_state_siz_u], mean=0, stddev=STD))
        self.br_u = tf.Variable(tf.zeros([self.hidden_state_siz_u]))
        self.bz_u = tf.Variable(tf.zeros([self.hidden_state_siz_u]))
        self.Wh_u = tf.Variable(
            tf.truncated_normal([self.hidden_state_siz_u, self.hidden_state_siz_u], mean=0, stddev=STD))

        # Weights for GRU_v
        self.Wx_v = tf.Variable(
            tf.truncated_normal([self.gru_input_siz_v + 2, self.hidden_state_siz_v], mean=0, stddev=STD))
        self.Wr_v = tf.Variable(
            tf.truncated_normal([self.gru_input_siz_v + 2, self.hidden_state_siz_v], mean=0, stddev=STD))
        self.Wz_v = tf.Variable(
            tf.truncated_normal([self.gru_input_siz_v + 2, self.hidden_state_siz_v], mean=0, stddev=STD))
        self.br_v = tf.Variable(tf.zeros([self.hidden_state_siz_v]))
        self.bz_v = tf.Variable(tf.zeros([self.hidden_state_siz_v]))
        self.Wh_v = tf.Variable(
            tf.truncated_normal([self.hidden_state_siz_v, self.hidden_state_siz_v], mean=0, stddev=STD))

        # Decoder GRU
        self.Wo_u_1 = tf.Variable(
            tf.truncated_normal([self.hidden_state_siz_u + self.gru_input_siz_u + 2, self.siz_u], mean=0, stddev=STD))
        self.Wo_u_2 = tf.Variable(
            tf.truncated_normal([self.hidden_state_siz_u + self.gru_input_siz_u + 2, self.siz_u], mean=0, stddev=STD))
        self.bo_u_1 = tf.Variable(tf.zeros([self.siz_u]))
        self.bo_u_2 = tf.Variable(tf.zeros([self.siz_u]))

        self.Wo_u = tf.Variable(tf.truncated_normal([self.hidden_state_siz_u, self.siz_u], mean=0, stddev=STD))
        self.bo_u = tf.Variable(tf.zeros([self.siz_u]))
        self.Wo_u_mu = tf.Variable(tf.truncated_normal([self.siz_u, self.siz_u_mu], mean=0, stddev=STD))
        self.bo_u_mu = tf.Variable(tf.zeros([self.siz_u_mu]))
        self.Wo_u_sig = tf.Variable(tf.truncated_normal([self.siz_u, self.siz_u_sig], mean=0, stddev=STD))
        self.bo_u_sig = tf.Variable(tf.zeros([self.siz_u_sig]))

        self.Wo_v_1 = tf.Variable(
            tf.truncated_normal([self.hidden_state_siz_v + self.gru_input_siz_v + 2, self.siz_v], mean=0, stddev=STD))
        self.Wo_v_2 = tf.Variable(
            tf.truncated_normal([self.hidden_state_siz_v + self.gru_input_siz_v + 2, self.siz_v], mean=0, stddev=STD))
        self.bo_v_1 = tf.Variable(tf.zeros([self.siz_v]))
        self.bo_v_2 = tf.Variable(tf.zeros([self.siz_v]))

        self.Wo_v = tf.Variable(tf.truncated_normal([self.hidden_state_siz_v, self.siz_v], mean=0, stddev=STD))
        self.bo_v = tf.Variable(tf.zeros([self.siz_v]))
        self.Wo_v_mu = tf.Variable(tf.truncated_normal([self.siz_v, self.siz_v_mu], mean=0, stddev=STD))
        self.bo_v_mu = tf.Variable(tf.zeros([self.siz_v_mu]))
        self.Wo_v_sig = tf.Variable(tf.truncated_normal([self.siz_v, self.siz_v_sig], mean=0, stddev=STD))
        self.bo_v_sig = tf.Variable(tf.zeros([self.siz_v_sig]))

        # Weights of MLP for generating Mu (rating mean)
        self.W_gen_mu_1 = tf.Variable(
            tf.truncated_normal([self.siz_u_mu + self.siz_v_mu, 16], mean=0, stddev=STD))
        self.b_gen_mu_1 = tf.Variable(tf.zeros([16]))

        self.W_gen_mu_2 = tf.Variable(
            tf.truncated_normal([16, 8], mean=0, stddev=STD))
        self.b_gen_mu_2 = tf.Variable(tf.zeros([8]))

        # Weights of MLP for generating Sigma (rating variance)
        self.W_gen_1 = tf.Variable(tf.truncated_normal([self.siz_u_mu + self.siz_v_mu,
                                                        self.siz_u_mu + self.siz_v_mu],
                                                       mean=0, stddev=STD))
        self.b_gen_1 = tf.Variable(tf.zeros(
            [self.siz_u_mu + self.siz_v_mu]))

        self.W_gen_2 = tf.Variable(tf.truncated_normal([self.siz_u_mu + self.siz_v_mu
                                                        + self.hidden_state_siz_u + self.hidden_state_siz_v,
                                                        self.siz_u_mu + self.siz_v_mu],
                                                       mean=0, stddev=STD))
        self.b_gen_2 = tf.Variable(tf.zeros([self.siz_u_mu + self.siz_v_mu]))

        self.W_gen_sig_1 = tf.Variable(tf.truncated_normal([self.siz_u_mu + self.siz_v_mu, 16], mean=0, stddev=STD))
        self.b_gen_sig_1 = tf.Variable(tf.zeros([16]))

        self.W_gen_sig_2 = tf.Variable(tf.truncated_normal([16, 8], mean=0, stddev=STD))
        self.b_gen_sig_2 = tf.Variable(tf.zeros([8]))

        # Weights of MLP for prior_u
        self.W_pri_u_1 = tf.Variable(tf.truncated_normal([self.hidden_state_siz_u + 2, self.siz_u], mean=0, stddev=STD))
        self.b_pri_u_1 = tf.Variable(tf.zeros([self.siz_u]))
        self.W_pri_u_2 = tf.Variable(tf.truncated_normal([self.hidden_state_siz_u + 2, self.siz_u], mean=0, stddev=STD))
        self.b_pri_u_2 = tf.Variable(tf.zeros([self.siz_u]))

        self.W_pri_u_mu = tf.Variable(tf.truncated_normal([self.siz_u, self.siz_u_mu], mean=0, stddev=STD))
        self.b_pri_u_mu = tf.Variable(tf.zeros([self.siz_u_mu]))
        self.W_pri_u_sig = tf.Variable(tf.truncated_normal([self.siz_u, self.siz_u_sig], mean=0, stddev=STD))
        self.b_pri_u_sig = tf.Variable(tf.zeros([self.siz_u_sig]))

        # Weights of MLP for prior_v
        self.W_pri_v_1 = tf.Variable(tf.truncated_normal(
            [self.hidden_state_siz_v + 2, self.siz_v], mean=0, stddev=STD))  # self.siz_h_v
        self.b_pri_v_1 = tf.Variable(tf.zeros(
            [self.siz_v]))

        self.W_pri_v_2 = tf.Variable(tf.truncated_normal(
            [self.hidden_state_siz_v + 2, self.siz_v], mean=0, stddev=STD))  # self.siz_h_v
        self.b_pri_v_2 = tf.Variable(tf.zeros(
            [self.siz_v]))

        self.W_pri_v_mu = tf.Variable(tf.truncated_normal(
            [self.siz_v, self.siz_v_mu], mean=0, stddev=STD))
        self.b_pri_v_mu = tf.Variable(tf.zeros(
            [self.siz_v_mu]))

        self.W_pri_v_sig = tf.Variable(tf.truncated_normal(
            [self.siz_v, self.siz_v_sig], mean=0, stddev=STD))
        self.b_pri_v_sig = tf.Variable(tf.zeros(
            [self.siz_v_sig]))

        ############################## Define Placeholders #############################

        # Placeholder for input vector with shape[batch, seq, embeddings]
        self.inputs_u = tf.sparse_placeholder(tf.float32)
        self.inputs_u_res_1 = tf.sparse_placeholder(tf.float32)
        self.inputs_u_res = tf.sparse_tensor_to_dense(self.inputs_u_res_1, validate_indices=False)

        self.inputs_v = tf.sparse_placeholder(tf.float32)
        self.inputs_v_res_1 = tf.sparse_placeholder(tf.float32)
        self.inputs_v_res = tf.sparse_tensor_to_dense(self.inputs_v_res_1, validate_indices=False)

        self.re_sp_u_ids = tf.placeholder(tf.int64, shape=[None, 2])
        self.u_seg_ids = tf.placeholder(tf.int64, shape=[None])

        self.re_sp_v_ids = tf.placeholder(tf.int64, shape=[None, 2])
        self.v_seg_ids = tf.placeholder(tf.int64, shape=[None])

        # Placeholder for initial state
        self.h_U = tf.placeholder(tf.float32, shape=[None, self.hidden_state_siz_u], name='h_U')
        self.h_V = tf.placeholder(tf.float32, shape=[None, self.hidden_state_siz_v], name='h_V')
        self.init_pre_u_mu = tf.placeholder(tf.float32, shape=[None, self.siz_u_mu], name='init_pre_u_mu')
        self.init_pre_v_mu = tf.placeholder(tf.float32, shape=[None, self.siz_v_mu], name='init_pre_v_mu')
        self.init_pre_u_sig = tf.placeholder(tf.float32, shape=[None, self.siz_u_sig], name='init_pre_u_sig')
        self.init_pre_v_sig = tf.placeholder(tf.float32, shape=[None, self.siz_v_sig], name='init_pre_v_sig')
        self.init_pre_u_sample = tf.placeholder(tf.float32, shape=[None, self.siz_u_mu], name='init_pre_u_sample')
        self.init_pre_v_sample = tf.placeholder(tf.float32, shape=[None, self.siz_v_mu], name='init_pre_v_sample')

        # Placeholder for mapped (user, item) pair with shape [batch num of ratings, 2]
        self.inputs_idx_pair = tf.placeholder(tf.int32, shape=[None, 4], name='idx_pair')

        # Placeholder for testing
        self.test_h_u = tf.placeholder(tf.float32, shape=[None, self.siz_h_u], name='test_h_u')
        self.test_h_v = tf.placeholder(tf.float32, shape=[None, self.siz_h_v], name='test_h_v')
        self.test_hidden_u = tf.placeholder(tf.float32, shape=[None, self.hidden_state_siz_u], name='test_hidden_u')
        self.test_hidden_v = tf.placeholder(tf.float32, shape=[None, self.hidden_state_siz_v], name='test_hidden_v')
        self.test_res = tf.placeholder(tf.float32, shape=[None, 5], name='test_res')

        # Placeholder for Ratings
        self.ratings = tf.placeholder(tf.float32, shape=[None], name='ratings')
        self.inputs_u_idx = tf.placeholder(tf.int32, shape=[None], name='inputs_u_idx')
        self.inputs_v_idx = tf.placeholder(tf.int32, shape=[None], name='inputs_v_idx')

    # Function for MLP to model GP prior of User
    def prior_u(self, pre_u, res, mark_new_u):
        """
        MLP to Update Prior Info. of User
        """
        c = tf.concat([pre_u, res, mark_new_u], 2)
        siz = tf.shape(c)
        c = tf.reshape(c, [siz[0] * siz[1], siz[2]])
        c_1 = tf.tanh(tf.matmul(c, self.W_pri_u_1) + self.b_pri_u_1)
        c_2 = tf.nn.relu(tf.matmul(c, self.W_pri_u_2) + self.b_pri_u_2)
        cur_u_mu = tf.matmul(c_1, self.W_pri_u_mu) + self.b_pri_u_mu
        cur_u_mu = tf.reshape(cur_u_mu, [siz[0], siz[1], self.siz_u_mu])
        cur_u_sig = tf.nn.softplus(tf.matmul(c_2, self.W_pri_u_sig) + self.b_pri_u_sig)
        cur_u_sig = tf.reshape(cur_u_sig, [siz[0], siz[1], self.siz_u_sig])

        return tf.concat([cur_u_mu, cur_u_sig], 2)

    # Function for MLP to model GP prior of Item
    def prior_v(self, pre_v, res, mark_new_v):
        """
        MLP to Update Prior Info. of Item
        """
        c = tf.concat([pre_v, res, mark_new_v], 2)
        siz = tf.shape(c)
        c = tf.reshape(c, [siz[0] * siz[1], siz[2]])
        c_1 = tf.tanh(tf.matmul(c, self.W_pri_v_1) + self.b_pri_v_1)
        c_2 = tf.nn.relu(tf.matmul(c, self.W_pri_v_2) + self.b_pri_v_2)
        cur_v_mu = tf.matmul(c_1, self.W_pri_v_mu) + self.b_pri_v_mu
        cur_v_mu = tf.reshape(cur_v_mu, [siz[0], siz[1], self.siz_v_mu])
        cur_v_sig = tf.nn.softplus(tf.matmul(c_2, self.W_pri_v_sig) + self.b_pri_v_sig)
        cur_v_sig = tf.reshape(cur_v_sig, [siz[0], siz[1], self.siz_v_sig])

        return tf.concat([cur_v_mu, cur_v_sig], 2)

    # Embedding unfixed length input into fixed length input of GRU
    def embedding_input_u(self):
        """
        Embedding input_u to GRU input x_u.
        """

        siz = self.inputs_u.dense_shape
        idx = self.inputs_u.indices
        val = self.inputs_u.values
        x_u = tf.transpose(tf.multiply(tf.transpose(tf.nn.embedding_lookup(self.W_embd_u_1, idx[:, -1])), val))
        x_u = tf.segment_sum(x_u, self.u_seg_ids)
        uu = tf.unstack(x_u, axis=1)
        x_u = tf.stack(
            [tf.sparse_to_dense(sparse_indices=self.re_sp_u_ids, output_shape=siz[0:2], sparse_values=x) for x in uu],
            axis=2)
        x_u = tf.concat([x_u,
                         tf.log(
                             tf.expand_dims(self.inputs_u_res[:, :, -2] - self.inputs_u_res[:, :, -3], axis=2) + 1e-10
                         ),
                         tf.expand_dims(self.inputs_u_res[:, :, -1], axis=2)], axis=2)
        return tf.tanh(x_u)

    def embedding_input_v(self):
        """
        Embedding input_v to GRU input x_v.
        """
        siz = self.inputs_v.dense_shape
        idx = self.inputs_v.indices
        val = self.inputs_v.values
        x_v = tf.transpose(tf.multiply(tf.transpose(tf.nn.embedding_lookup(self.W_embd_v_1, idx[:, -1])), val))
        x_v = tf.segment_sum(x_v, self.v_seg_ids)
        vv = tf.unstack(x_v, axis=1)
        x_v = tf.stack(
            [tf.sparse_to_dense(sparse_indices=self.re_sp_v_ids, output_shape=siz[0:2], sparse_values=x) for x in vv],
            axis=2)
        x_v = tf.concat([x_v,
                         tf.log(
                             tf.expand_dims(self.inputs_v_res[:, :, -2] - self.inputs_v_res[:, :, -3], axis=2) + 1e-10
                         ),
                         tf.expand_dims(self.inputs_v_res[:, :, -1], axis=2)], axis=2)
        return tf.tanh(x_v)

    # Function for User GRU cell
    def gru_u(self, previous_hidden_memory_tuple, x_u):
        """
        User GRU Equations
        """
        pre_h_u = previous_hidden_memory_tuple[0]

        pre_u_mu, pre_u_sig = self.get_output_u_mu(pre_h_u, x_u), self.get_output_u_sig(pre_h_u, x_u)

        eps = tf.random_normal([self.siz_u_mu], 0.0, 1.0, dtype=tf.float32)
        pre_u_sample = pre_u_mu + tf.multiply(pre_u_sig, eps)

        z = tf.sigmoid(tf.matmul(x_u, self.Wz_u) + self.bz_u)
        r = tf.sigmoid(tf.matmul(x_u, self.Wr_u) + self.br_u)

        tmp = tf.exp(-(tf.exp(x_u[0, self.gru_input_siz_u] - 1e-10)) / self.interval_u)
        pre_h_u = tmp * pre_h_u

        h_ = tf.tanh(tf.matmul(x_u, self.Wx_u) + tf.matmul(pre_h_u, self.Wh_u) * r)

        h_u = tf.multiply((1 - z), h_) + tf.multiply(pre_h_u, z)
        return [h_u, pre_u_mu, pre_u_sig, pre_u_sample, pre_h_u]

    # Function for Item GRU cell
    def gru_v(self, previous_hidden_memory_tuple, x_v):
        """
        Item GRU Equations
        """
        pre_h_v = previous_hidden_memory_tuple[0]

        pre_v_mu, pre_v_sig = self.get_output_v_mu(pre_h_v, x_v), self.get_output_v_sig(pre_h_v, x_v)

        eps = tf.random_normal([self.siz_v_mu], 0.0, 1.0, dtype=tf.float32)
        pre_v_sample = pre_v_mu + tf.multiply(pre_v_sig, eps)

        z = tf.sigmoid(tf.matmul(x_v, self.Wz_v) + self.bz_v)
        r = tf.sigmoid(tf.matmul(x_v, self.Wr_v) + self.br_v)

        tmp = tf.exp(-(tf.exp(x_v[0, self.gru_input_siz_v] - 1e-10)) / self.interval_v)
        pre_h_v = tmp * pre_h_v

        h_ = tf.tanh(tf.matmul(x_v, self.Wx_v) + tf.matmul(pre_h_v, self.Wh_v) * r)

        h_v = tf.multiply((1 - z), h_) + tf.multiply(pre_h_v, z)

        return [h_v, pre_v_mu, pre_v_sig, pre_v_sample, pre_h_v]

    # Function of the Generative Networks
    def generative_mu(self, u, v):
        """
        Generative ratings.
        """
        # MF part
        mf_mu = tf.reduce_sum(tf.multiply(u, v), 1)

        # MLP part
        c = tf.concat([u, v], 1)
        tmp = tf.nn.relu(tf.matmul(c, self.W_gen_1) + self.b_gen_1)
        tmp = tf.nn.relu(tf.matmul(tmp, self.W_gen_mu_1) + self.b_gen_mu_1)
        mlp_mu = tf.reduce_sum(tf.matmul(tmp, self.W_gen_mu_2) + self.b_gen_mu_2, 1)

        return self.min_rating + (self.max_rating - self.min_rating) * (tf.sigmoid(mf_mu + mlp_mu))

    def generative_sigma2(self, u, v, h_u, h_v):
        """
        Generative ratings.
        """
        c = tf.concat([u, v, h_u, h_v], 1)
        c = tf.nn.relu(tf.matmul(c, self.W_gen_2) + self.b_gen_2)
        tmp = tf.nn.relu(tf.matmul(c, self.W_gen_sig_1) + self.b_gen_sig_1)
        sigma = tf.reduce_sum(tf.nn.softplus(tf.matmul(tmp, self.W_gen_sig_2) + self.b_gen_sig_2), 1)

        return sigma

    # Function for getting all hidden state.
    def get_states_u(self):
        """
        Iterates through time/ sequence to get all hidden state 
        """
        all_hidden_states_u = tf.scan(self.gru_u,
                                      process_batch_input_for_rnn(self.embedding_input_u()),
                                      initializer=[self.h_U, self.init_pre_u_mu, self.init_pre_u_sig,
                                                   self.init_pre_u_sample, self.h_U],
                                      name='states')
        return all_hidden_states_u

    # Function for getting all hidden state.
    def get_states_v(self):
        """
        Iterates through time/ sequence to get all hidden state
        """
        all_hidden_states_v = tf.scan(self.gru_v,
                                      process_batch_input_for_rnn(self.embedding_input_v()),
                                      initializer=[self.h_V, self.init_pre_v_mu, self.init_pre_v_sig,
                                                   self.init_pre_v_sample, self.h_V],
                                      name='states')
        return all_hidden_states_v

    # Function to get output from a hidden layer
    def get_output_u_mu(self, h, x):
        """
        This function takes hidden state and returns output
        """
        hidden_state = tf.concat([h, x], axis=1)
        output = tf.nn.relu(tf.matmul(hidden_state, self.Wo_u_1) + self.bo_u_1)
        output = tf.matmul(output, self.Wo_u_mu) + self.bo_u_mu
        return output

    def get_output_u_sig(self, h, x):
        """
        This function takes hidden state and returns output
        """
        hidden_state = tf.concat([h, x], axis=1)
        output = tf.nn.relu(tf.matmul(hidden_state, self.Wo_u_2) + self.bo_u_2)
        output = tf.nn.softplus(tf.matmul(output, self.Wo_u_sig) + self.bo_u_sig)
        return output

    # Function to get output from a hidden layer
    def get_output_v_mu(self, h, x):
        """
        This function takes hidden state and returns output
        """
        hidden_state = tf.concat([h, x], axis=1)
        output = tf.nn.relu(tf.matmul(hidden_state, self.Wo_v_1) + self.bo_v_1)
        output = tf.matmul(output, self.Wo_v_mu) + self.bo_v_mu
        return output

    def get_output_v_sig(self, h, x):
        """
        This function takes hidden state and returns output
        """
        hidden_state = tf.concat([h, x], axis=1)
        output = tf.nn.relu(tf.matmul(hidden_state, self.Wo_v_2) + self.bo_v_2)
        output = tf.nn.softplus(tf.matmul(output, self.Wo_v_sig) + self.bo_v_sig)
        return output

    def predict_ratings(self):

        [_, _, _, _, pre_hidden_us] = self.get_states_u()
        [_, _, _, _, pre_hidden_vs] = self.get_states_v()

        pre_hidden_us = process_batch_input_for_rnn(pre_hidden_us)
        pre_hidden_vs = process_batch_input_for_rnn(pre_hidden_vs)

        p_pri_u = self.prior_u(pre_hidden_us,
                               tf.log(tf.expand_dims(self.inputs_u_res[:, :, -2] - self.inputs_u_res[:, :, -3],
                                                     2) + 1e-10),
                               tf.expand_dims(self.inputs_u_res[:, :, -1], 2))

        p_pri_v = self.prior_v(pre_hidden_vs,
                               tf.log(tf.expand_dims(self.inputs_v_res[:, :, -2] - self.inputs_v_res[:, :, -3],
                                                     2) + 1e-10),
                               tf.expand_dims(self.inputs_v_res[:, :, -1], 2))

        # Update p_u, p_v
        p_u_mu = p_pri_u[:, :, 0:self.siz_u_mu]
        p_S_u = p_pri_u[:, :, self.siz_u_mu:]
        p_u_sig = tf.abs(p_S_u)

        p_v_mu = p_pri_v[:, :, 0:self.siz_v_mu]
        p_S_v = p_pri_v[:, :, self.siz_v_mu:]
        p_v_sig = tf.abs(p_S_v)

        u_idx = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.inputs_idx_pair), tf.constant([0, 2])))
        v_idx = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.inputs_idx_pair), tf.constant([1, 3])))

        sample_p_u_mu = tf.gather_nd(p_u_mu, u_idx)
        sample_p_u_sig = tf.gather_nd(p_u_sig, u_idx)
        sample_p_v_mu = tf.gather_nd(p_v_mu, v_idx)
        sample_p_v_sig = tf.gather_nd(p_v_sig, v_idx)

        st_u_idx = tf.gather_nd(self.inputs_u_idx, tf.expand_dims(self.inputs_idx_pair[:, 0], 1))
        sample_p_u_st = tf.gather_nd(self.U_st, tf.expand_dims(st_u_idx, 1))
        sample_p_u_mu += sample_p_u_st

        st_v_idx = tf.gather_nd(self.inputs_v_idx, tf.expand_dims(self.inputs_idx_pair[:, 1], 1))
        sample_p_v_st = tf.gather_nd(self.V_st, tf.expand_dims(st_v_idx, 1))
        sample_p_v_mu += sample_p_v_st

        pre_h_u = tf.gather_nd(pre_hidden_us, u_idx)
        pre_h_v = tf.gather_nd(pre_hidden_vs, v_idx)

        predictions = self.generative_mu(sample_p_u_mu, sample_p_v_mu)
        sig = self.generative_sigma2(sample_p_u_mu, sample_p_v_mu, pre_h_u, pre_h_v)

        return {'mean': predictions,
                'var': sig}

    def train(self):
        """
        Iterating through hidden states to get outputs for all timestamp
        """
        # Get hidden states of current batch of users and movies
        [hidden_us, pre_u_mu, pre_u_sig, pre_u_sample, pre_hidden_us] = self.get_states_u()
        [hidden_vs, pre_v_mu, pre_v_sig, pre_v_sample, pre_hidden_vs] = self.get_states_v()

        hidden_us = process_batch_input_for_rnn(hidden_us)
        pre_hidden_us = process_batch_input_for_rnn(pre_hidden_us)
        pre_u_mu = process_batch_input_for_rnn(pre_u_mu)
        pre_u_sig = process_batch_input_for_rnn(pre_u_sig)
        pre_u_sample = process_batch_input_for_rnn(pre_u_sample)

        hidden_vs = process_batch_input_for_rnn(hidden_vs)
        pre_hidden_vs = process_batch_input_for_rnn(pre_hidden_vs)
        pre_v_mu = process_batch_input_for_rnn(pre_v_mu)
        pre_v_sig = process_batch_input_for_rnn(pre_v_sig)
        pre_v_sample = process_batch_input_for_rnn(pre_v_sample)

        # Calculate Variational Distributions for every timestamp
        with tf.variable_scope('variational'):
            # Update qs
            q_u_mu = pre_u_mu
            q_S_u = pre_u_sig
            q_u_sig = tf.maximum(1e-10, tf.abs(q_S_u))

            q_v_mu = pre_v_mu
            q_S_v = pre_v_sig
            q_v_sig = tf.maximum(1e-10, tf.abs(q_S_v))

        # Calculate priors
        with tf.variable_scope('prior'):
            p_pri_u = self.prior_u(pre_hidden_us,
                                   tf.log(tf.expand_dims(self.inputs_u_res[:, :, -2] - self.inputs_u_res[:, :, -3],
                                                         2) + 1e-10),
                                   tf.expand_dims(self.inputs_u_res[:, :, -1], 2))

            p_pri_v = self.prior_v(pre_hidden_vs,
                                   tf.log(tf.expand_dims(self.inputs_v_res[:, :, -2] - self.inputs_v_res[:, :, -3],
                                                         2) + 1e-10),
                                   tf.expand_dims(self.inputs_v_res[:, :, -1], 2))

            # Update p_u, p_v
            p_u_mu = p_pri_u[:, :, 0:self.siz_u_mu]
            p_S_u = p_pri_u[:, :, self.siz_u_mu:]
            p_u_sig = tf.abs(p_S_u)

            p_v_mu = p_pri_v[:, :, 0:self.siz_v_mu]
            p_S_v = p_pri_v[:, :, self.siz_v_mu:]
            p_v_sig = tf.abs(p_S_v)

        def tf_kl_gaussgauss(mu_1, sig_1, mu_2, sig_2):
            with tf.variable_scope("kl_gaussgauss"):
                return tf.reduce_sum(0.5 * (
                        2 * tf.log(tf.maximum(1e-10, sig_2), name='log_sig_2')
                        - 2 * tf.log(tf.maximum(1e-10, sig_1), name='log_sig_1')
                        + (tf.square(sig_1) + tf.square(mu_1 - mu_2)) / tf.maximum(1e-10, (tf.square(sig_2))) - 1
                ), 1)

        def tf_normal(y, mu, s):
            with tf.variable_scope('normal'):
                ss = tf.maximum(1e-10, tf.square(s))
                norm = tf.square(tf.subtract(y, mu))
                z = tf.div(norm, ss)
                denom_log = tf.log(ss, name='denom_log')
                # result = -tf.reduce_sum(z + denom_log) / 2
                result = -(z + denom_log) / 2
            return result

        # Generate xs
        with tf.variable_scope('generative'):
            # The likelihood is Normal distributed with Mu and Sigma2 given by the
            # generative network
            # Recursive all the embeddings to generate the predicted ratings

            u_idx = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.inputs_idx_pair), tf.constant([0, 2])))
            v_idx = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.inputs_idx_pair), tf.constant([1, 3])))
            u_sample = tf.gather_nd(pre_u_sample, u_idx)
            v_sample = tf.gather_nd(pre_v_sample, v_idx)
            pre_h_u = tf.gather_nd(pre_hidden_us, u_idx)
            pre_h_v = tf.gather_nd(pre_hidden_vs, v_idx)

            sample_q_u_mu = tf.gather_nd(q_u_mu, u_idx)
            sample_q_u_sig = tf.gather_nd(q_u_sig, u_idx)
            sample_p_u_mu = tf.gather_nd(p_u_mu, u_idx)
            sample_p_u_sig = tf.gather_nd(p_u_sig, u_idx)

            st_u_idx = tf.gather_nd(self.inputs_u_idx, tf.expand_dims(self.inputs_idx_pair[:, 0], 1))
            sample_p_u_st = tf.gather_nd(self.U_st, tf.expand_dims(st_u_idx, 1))
            sample_p_u_mu += sample_p_u_st

            sample_q_v_mu = tf.gather_nd(q_v_mu, v_idx)
            sample_q_v_sig = tf.gather_nd(q_v_sig, v_idx)
            sample_p_v_mu = tf.gather_nd(p_v_mu, v_idx)
            sample_p_v_sig = tf.gather_nd(p_v_sig, v_idx)

            st_v_idx = tf.gather_nd(self.inputs_v_idx, tf.expand_dims(self.inputs_idx_pair[:, 1], 1))
            sample_p_v_st = tf.gather_nd(self.V_st, tf.expand_dims(st_v_idx, 1))
            sample_p_v_mu += sample_p_v_st

            u_sample += sample_p_u_st
            v_sample += sample_p_v_st
            sample_q_u_mu += sample_p_u_st
            sample_q_v_mu += sample_p_v_st

            mu = self.generative_mu(u_sample, v_sample)
            sig = self.generative_sigma2(u_sample, v_sample, pre_h_u, pre_h_v)
            kl_u = tf_kl_gaussgauss(sample_q_u_mu, sample_q_u_sig, sample_p_u_mu, sample_p_u_sig)
            kl_v = tf_kl_gaussgauss(sample_q_v_mu, sample_q_v_sig, sample_p_v_mu, sample_p_v_sig)

            expected_log_likelihood = tf.constant(0.0)
            expected_log_likelihood += tf_normal(tf.cast(self.ratings, tf.float32), mu, sig)

            # U_st, V_st, Prior
            expected_log_likelihood -= 1e-3 * tf.reduce_sum(tf.square(sample_p_u_st), 1)
            expected_log_likelihood -= 1e-3 * tf.reduce_sum(tf.square(sample_p_v_st), 1)
            elbo = tf.reduce_mean(expected_log_likelihood - kl_u - kl_v)

        return {'elbo': elbo, 'hidden_us': hidden_us, 'hidden_vs': hidden_vs}
