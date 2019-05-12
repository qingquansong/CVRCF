import os
import GPUtil
import time
import argparse
import numpy as np
import tensorflow as tf
import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from dataloader import DataSet
from model import CVRCF

parser = argparse.ArgumentParser(description='MovieLens-10M CVRCF experiment')

# dataset hyperparameters.
parser.add_argument('--num_of_u', type=int, default=71567, help='num_of_users')
parser.add_argument('--num_of_v', type=int, default=10681, help='num_of_items')
parser.add_argument('--rating_upper_bound', type=int, default=5, help='rating_upper_bound')
parser.add_argument('--rating_lower_bound', type=int, default=0, help='rating_lower_bound')

# experimental hyperparameters.
parser.add_argument('--lr', type=float, default=1e-3, help='learning_rate')
parser.add_argument('--n_epoch', type=int, default=3, help='num_of_training_epoch (used only in training phase)')
parser.add_argument('--time_interval', type=int, default=20, help='training_batch_granularity (# of weeks)')
parser.add_argument('--test_time_interval', type=int, default=4, help='testing_batch_granularity (# of weeks)')
parser.add_argument('--gran_u', type=int, default=4, help='user_training_granularity (# of weeks)')
parser.add_argument('--gran_v', type=int, default=4, help='item_training_granularity (# of weeks)')
parser.add_argument('--test_gran_u', type=int, default=4, help='user_testing_granularity (# of weeks)')
parser.add_argument('--test_gran_v', type=int, default=4, help='item_testing_granularity (# of weeks)')
parser.add_argument('--max_batch_size', type=int, default=10000000, help='limit batch size when gran_u/v are too big')
parser.add_argument('--max_t', type=int, default=50, help='max_model_update_iterations_per_training_granularity')
parser.add_argument('--test_max_t', type=int, default=50, help='max_model_update_iterationss_per_testing_granularity')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')

# model architecture hyperparameters
# currently implementation requires: for either user or item the stationary factor and dynamic factors
# should be the same, it is easy to modify them to be different via adding additional embedding layers
parser.add_argument('--siz_u_st', type=int, default=20, help='user stationary factor size')
parser.add_argument('--siz_v_st', type=int, default=20, help='item stationary factor size')
parser.add_argument('--siz_u_dy', type=int, default=20, help='user dynamic factor size')
parser.add_argument('--siz_v_dy', type=int, default=20, help='item dynamic factor size')
parser.add_argument('--gru_input_siz_u', type=int, default=40, help='input embedding size of user GRU')
parser.add_argument('--gru_input_siz_v', type=int, default=40, help='input embedding size of item GRU')
parser.add_argument('--hidden_state_siz_u', type=int, default=20, help='user GRU hidden state size')
parser.add_argument('--hidden_state_siz_v', type=int, default=20, help='item GRU hidden state size')
parser.add_argument('--interval_u', type=int, default=1, help='user_exponential_decay_interval (# of days)')
parser.add_argument('--interval_v', type=int, default=4, help='item_exponential_decay_interval (# of days)')


def select_gpu():
    try:
        # Get the first available GPU
        device_id_list = GPUtil.getFirstAvailable()
        device_id = device_id_list[0]  # grab first element from list

        # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    except EnvironmentError:
        print("GPU not found")


def test_one_step(test_data, model, sess, pred_all, hidden_U, hidden_V, 
                  MSE=None, RMSE=None, MSE1=None, RMSE1=None, N=None, N1=None):
    if test_data.finish != 1:
        test_sp_u_indices, test_sp_u_shape, test_sp_u_val, \
        test_sp_u_indices_res, test_sp_u_shape_res, test_sp_u_val_res, \
        test_sp_v_indices, test_sp_v_shape, test_sp_v_val, \
        test_sp_v_indices_res, test_sp_v_shape_res, test_sp_v_val_res, \
        test_inputs_u_idx, test_inputs_v_idx, \
        test_inputs_idx_pair, test_all_data = test_data.get_batch_data()

        print('Read Finish!\n')
        print(test_inputs_idx_pair.shape)
        print(np.max(test_inputs_idx_pair, 0))

        siz = test_all_data.shape
        mark_new_user_movie = np.zeros([siz[0], 2])

        tmp_u = np.concatenate((test_sp_u_indices, np.expand_dims(test_sp_u_val, axis=1)), axis=1)
        tmp_v = np.concatenate((test_sp_v_indices, np.expand_dims(test_sp_v_val, axis=1)), axis=1)
        tmp_u.view('i8,i8,i8,i8').sort(order=['f0', 'f1', 'f2'], axis=0)  #
        tmp_v.view('i8,i8,i8,i8').sort(order=['f0', 'f1', 'f2'], axis=0)  #
        test_sp_u_indices, test_sp_u_val = tmp_u[:, 0:3], tmp_u[:, 3]
        test_sp_v_indices, test_sp_v_val = tmp_v[:, 0:3], tmp_v[:, 3]
        test_re_sp_u_ids, test_u_seg_ids = np.unique(test_sp_u_indices[:, 0:2], axis=0, return_inverse=True)
        test_re_sp_v_ids, test_v_seg_ids = np.unique(test_sp_v_indices[:, 0:2], axis=0, return_inverse=True)

        pred_ratings, sig = sess.run([pred_all['mean'], pred_all['var']],
                                     feed_dict={
                                         model.inputs_u: (test_sp_u_indices, test_sp_u_val, test_sp_u_shape),
                                         model.inputs_v: (test_sp_v_indices, test_sp_v_val, test_sp_v_shape),
                                         model.inputs_u_res_1: (
                                             test_sp_u_indices_res, test_sp_u_val_res, test_sp_u_shape_res),
                                         model.inputs_v_res_1: (
                                             test_sp_v_indices_res, test_sp_v_val_res, test_sp_v_shape_res),
                                         model.re_sp_u_ids: test_re_sp_u_ids,
                                         model.u_seg_ids: test_u_seg_ids,
                                         model.re_sp_v_ids: test_re_sp_v_ids,
                                         model.v_seg_ids: test_v_seg_ids,
                                         model.h_U: hidden_U[test_inputs_u_idx, :],
                                         model.h_V: hidden_V[test_inputs_v_idx, :],
                                         model.inputs_u_idx: test_inputs_u_idx,
                                         model.inputs_v_idx: test_inputs_v_idx,
                                         model.inputs_idx_pair: test_inputs_idx_pair[:, 0:4],
                                         model.ratings: test_inputs_idx_pair[:, 4]})

        qq0, qq1 = [], []
        for i in range(siz[0]):
            if int(test_all_data[i, 4]) == 0:
                mark_new_user_movie[i, 0] = 1
                qq0.append(test_all_data[i, 0])
            if int(test_all_data[i, 5]) == 0:
                mark_new_user_movie[i, 1] = 1
                qq1.append(test_all_data[i, 1])

        for i in range(siz[0]):
            if test_all_data[i, 0] in qq0:
                mark_new_user_movie[i, 0] = 1
            if test_all_data[i, 1] in qq1:
                mark_new_user_movie[i, 1] = 1

        real_ratings = test_all_data[:, 2] * (1 - mark_new_user_movie[:, 0]) * (1 - mark_new_user_movie[:, 1])
        real_ratings1 = real_ratings[real_ratings != 0]
        pred_ratings1 = pred_ratings[real_ratings != 0]
        if MSE is not None:
            MSE.append(mean_squared_error(real_ratings1, pred_ratings1))
            RMSE.append(sqrt(MSE[-1]))
            N.append(len(real_ratings1))
            print(MSE[-1])
            print(RMSE[-1])
            np.save("results/N_ml_10M.npy", np.array(N))
            np.save("results/MSE_ml_10M.npy", np.array(MSE))
            np.save("results/RMSE_ml_10M.npy", np.array(RMSE))
        else:
            MSE = mean_squared_error(real_ratings1, pred_ratings1)
            RMSE = sqrt(MSE)
            print(MSE)
            print(RMSE)
            np.save("results/N1_ml_10M.npy", np.array(N1))
            np.save("results/MSE1_ml_10M.npy", np.array(MSE1))
            np.save("results/RMSE1_ml_10M.npy", np.array(RMSE1))
        print(real_ratings[0:20])
        print(pred_ratings[0:20])
        print('\n')

        real_ratings = test_all_data[:, 2]
        if MSE1 is not None:
            MSE1.append(mean_squared_error(real_ratings, pred_ratings))
            RMSE1.append(sqrt(MSE1))
            N1.append(len(real_ratings))
            print(MSE1[-1])
            print(RMSE1[-1])
        else:
            MSE1 = mean_squared_error(real_ratings, pred_ratings)
            RMSE1 = sqrt(MSE1)
            print(MSE1)
            print(RMSE1)
        print(real_ratings[0:20])
        print(pred_ratings[0:20])
        print('\n')
    return test_data


def testing_phase(model, sess, train_all, pred_all, mark_u_time_end, mark_v_time_end, hidden_U, hidden_V):
    MSE, RMSE, MSE1, RMSE1, N, N1 = [], [], [], [], [], []
    train_data = DataSet('data/test.txt', args, mark_u_time_end, mark_v_time_end)
    test_data = DataSet('data/test.txt', args, train_data.mark_u_time_end, train_data.mark_v_time_end)
    g = 0
    while test_data.finish != 1:
        g += 1
        print(g)

        # Test
        test_data = test_one_step(test_data, model, sess, pred_all, hidden_U, hidden_V, MSE, RMSE, MSE1, RMSE1, N, N1)

        # Updating
        sp_u_indices, sp_u_shape, sp_u_val, \
        sp_u_indices_res, sp_u_shape_res, sp_u_val_res, \
        sp_v_indices, sp_v_shape, sp_v_val, \
        sp_v_indices_res, sp_v_shape_res, sp_v_val_res, \
        inputs_u_idx, inputs_v_idx, \
        inputs_idx_pair, all_data = train_data.get_batch_data()

        print('Read Finish!\n')
        print(inputs_idx_pair.shape)
        print(np.max(inputs_idx_pair, 0))

        tmp_u = np.concatenate((sp_u_indices, np.expand_dims(sp_u_val, axis=1)), axis=1)
        tmp_v = np.concatenate((sp_v_indices, np.expand_dims(sp_v_val, axis=1)), axis=1)
        tmp_u.view('i8,i8,i8,i8').sort(order=['f0', 'f1', 'f2'], axis=0)  #
        tmp_v.view('i8,i8,i8,i8').sort(order=['f0', 'f1', 'f2'], axis=0)  #
        sp_u_indices, sp_u_val = tmp_u[:, 0:3], tmp_u[:, 3]
        sp_v_indices, sp_v_val = tmp_v[:, 0:3], tmp_v[:, 3]
        re_sp_u_ids, u_seg_ids = np.unique(sp_u_indices[:, 0:2], axis=0, return_inverse=True)
        re_sp_v_ids, v_seg_ids = np.unique(sp_v_indices[:, 0:2], axis=0, return_inverse=True)

        loss0, loss = 0, 100
        t = 0
        while abs(loss - loss0) / abs(loss) > 1e-2 and t < args.test_max_t:
            t += 1
            print('qq')
            print(t)
            print('qq')
            _, loss, hidden_us, hidden_vs \
                = sess.run([train_all['train_op'], train_all['elbo'], train_all['hidden_us'], train_all['hidden_vs']],
                           feed_dict={model.inputs_u: (sp_u_indices, sp_u_val, sp_u_shape),
                                      model.inputs_v: (sp_v_indices, sp_v_val, sp_v_shape),
                                      model.inputs_u_res_1: (sp_u_indices_res, sp_u_val_res, sp_u_shape_res),
                                      model.inputs_v_res_1: (sp_v_indices_res, sp_v_val_res, sp_v_shape_res),
                                      model.re_sp_u_ids: re_sp_u_ids,
                                      model.u_seg_ids: u_seg_ids,
                                      model.re_sp_v_ids: re_sp_v_ids,
                                      model.v_seg_ids: v_seg_ids,
                                      model.h_U: hidden_U[inputs_u_idx, :],
                                      model.h_V: hidden_V[inputs_v_idx, :],
                                      model.inputs_u_idx: inputs_u_idx,
                                      model.inputs_v_idx: inputs_v_idx,
                                      model.inputs_idx_pair: inputs_idx_pair[:, 0:4],
                                      model.ratings: inputs_idx_pair[:, 4]})
            print('loss: {}'.format(loss))

        for i in range(inputs_u_idx.shape[0]):
            tmp_idx = int(max(inputs_idx_pair[inputs_idx_pair[:, 0] == i, 2]))
            hidden_U[inputs_u_idx[i], :] = hidden_us[i, tmp_idx, :]
        for j in range(inputs_v_idx.shape[0]):
            tmp_idx = int(max(inputs_idx_pair[inputs_idx_pair[:, 1] == j, 3]))
            hidden_V[inputs_v_idx[j], :] = hidden_vs[j, tmp_idx, :]


def main(args):
    # Select running device
    # select_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Initialize the latent factors
    hidden_U = np.zeros([args.num_of_u, args.hidden_state_siz_u])
    hidden_V = np.zeros([args.num_of_v, args.hidden_state_siz_v])

    model = CVRCF(args)

    # Define elbo loss and optimizer
    train_all = model.train()

    pred_all = model.predict_ratings()

    optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)  # AdamOptimizer
    train_all['train_op'] = optimizer.minimize(-train_all['elbo'])

    # Start training
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    # config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    init = tf.global_variables_initializer()

    # Run the initializer
    sess.run(init)

    # Model training
    for epoch in range(1, args.n_epoch + 1):
        dd = 0
        mark_u_time_end = np.zeros(args.num_of_u)
        mark_v_time_end = np.zeros(args.num_of_v)

        train_data = DataSet('data/train.txt', args, mark_u_time_end, mark_v_time_end)
        test_data = DataSet('data/train.txt', args, train_data.mark_u_time_end, train_data.mark_v_time_end)

        # Skip the first batch for testing purpose
        test_data.get_batch_data()

        print(epoch)
        start_time = time.time()
        while train_data.finish != 1:

            # Run optimization op (backprop)
            sp_u_indices, sp_u_shape, sp_u_val, \
            sp_u_indices_res, sp_u_shape_res, sp_u_val_res, \
            sp_v_indices, sp_v_shape, sp_v_val, \
            sp_v_indices_res, sp_v_shape_res, sp_v_val_res, \
            inputs_u_idx, inputs_v_idx, \
            inputs_idx_pair, all_data = train_data.get_batch_data()

            tmp_u = np.concatenate((sp_u_indices, np.expand_dims(sp_u_val, axis=1)), axis=1)
            tmp_v = np.concatenate((sp_v_indices, np.expand_dims(sp_v_val, axis=1)), axis=1)
            tmp_u.view('i8,i8,i8,i8').sort(order=['f0', 'f1', 'f2'], axis=0)  #
            tmp_v.view('i8,i8,i8,i8').sort(order=['f0', 'f1', 'f2'], axis=0)  #
            sp_u_indices, sp_u_val = tmp_u[:, 0:3], tmp_u[:, 3]
            sp_v_indices, sp_v_val = tmp_v[:, 0:3], tmp_v[:, 3]
            re_sp_u_ids, u_seg_ids = np.unique(sp_u_indices[:, 0:2], axis=0, return_inverse=True)
            re_sp_v_ids, v_seg_ids = np.unique(sp_v_indices[:, 0:2], axis=0, return_inverse=True)

            print('Read Finish!\n')
            dd += 1
            print(dd)

            print(inputs_idx_pair.shape)
            print(np.max(inputs_idx_pair, 0))

            loss0, loss = 0, 100
            t = 0
            while abs(loss - loss0) / abs(loss) > 1e-2 and t < args.max_t:
                t += 1
                print('qq')
                print(t)
                print('qq')
                _, loss, hidden_us, hidden_vs = sess.run([train_all['train_op'], train_all['elbo'],
                                                          train_all['hidden_us'], train_all['hidden_vs']],
                                                         feed_dict={
                                                             model.inputs_u: (sp_u_indices, sp_u_val, sp_u_shape),
                                                             model.inputs_v: (sp_v_indices, sp_v_val, sp_v_shape),
                                                             model.inputs_u_res_1:
                                                                 (sp_u_indices_res, sp_u_val_res, sp_u_shape_res),
                                                             model.inputs_v_res_1:
                                                                 (sp_v_indices_res, sp_v_val_res, sp_v_shape_res),
                                                             model.re_sp_u_ids: re_sp_u_ids,
                                                             model.u_seg_ids: u_seg_ids,
                                                             model.re_sp_v_ids: re_sp_v_ids,
                                                             model.v_seg_ids: v_seg_ids,
                                                             model.h_U: hidden_U[inputs_u_idx, :],
                                                             model.h_V: hidden_V[inputs_v_idx, :],
                                                             model.inputs_u_idx: inputs_u_idx,
                                                             model.inputs_v_idx: inputs_v_idx,
                                                             model.inputs_idx_pair: inputs_idx_pair[:, 0:4],
                                                             model.ratings: inputs_idx_pair[:, 4]})
                print('loss: {}'.format(loss))

            for i in range(inputs_u_idx.shape[0]):
                tmp_idx = int(max(inputs_idx_pair[inputs_idx_pair[:, 0] == i, 2]))
                hidden_U[inputs_u_idx[i], :] = hidden_us[i, tmp_idx, :]
            for j in range(inputs_v_idx.shape[0]):
                tmp_idx = int(max(inputs_idx_pair[inputs_idx_pair[:, 1] == j, 3]))
                hidden_V[inputs_v_idx[j], :] = hidden_vs[j, tmp_idx, :]

            # Test
            test_data = DataSet('data/test.txt', args, train_data.mark_u_time_end, train_data.mark_v_time_end)
            test_data = test_one_step(test_data, model, sess, pred_all, hidden_U, hidden_V)

        if not (train_data.finish == 0 or epoch == args.n_epoch):
            # Reinitialize to default value after finishing one epoch
            print("\n One Epoch Finished! \n")
            hidden_U = np.zeros([args.num_of_u, args.hidden_state_siz_u])
            hidden_V = np.zeros([args.num_of_v, args.hidden_state_siz_v])

        print("--- %s seconds ---" % (time.time() - start_time))

        test_data.finish = 0
        train_data.finish = 0
    print("Optimization Finished!")

    # Testing Phase
    testing_phase(model, sess, train_all, pred_all, mark_u_time_end, mark_v_time_end, hidden_U, hidden_V)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
