import logging
import os
from time import time
from typing import List

import numpy as np
import tensorflow as tf

CHECKPOINTS_DIR = "./checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR, "best_rmse_model.ckpt")
DATASET_PATH = "./dataset/"
LOG_PATH = "./model/model.log"

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs("./model", exist_ok=True)

n_hid = 500
n_dim = 5
n_layers = 2
gk_size = 3
lambda_2 = 70.
lambda_s = 0.018
iter_p = 30
iter_f = 30
epoch_p = 15
epoch_f = 15
dot_scale = 0.5

logging.basicConfig(filename="./model/model.log", level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


def load_data_1m(path=DATASET_PATH, delimiter='::'):
    tic = time()
    print('reading data...')

    # Загружаем заранее разделённые данные
    data_train = np.loadtxt(path + 'ratings_train.dat', skiprows=0, delimiter=delimiter).astype('int32')
    data_test = np.loadtxt(path + 'ratings_test.dat', skiprows=0, delimiter=delimiter).astype('int32')

    print('taken', time() - tic, 'seconds')

    # Объединяем для построения словарей пользователей и фильмов
    data_all = np.concatenate((data_train, data_test), axis=0)

    # Уникальные ID пользователей и фильмов
    all_users = np.unique(data_all[:, 0])
    all_movies = np.unique(data_all[:, 1])

    n_u = all_users.size
    n_m = all_movies.size

    # Создаём отображения ID в индексы
    udict = {u: i for i, u in enumerate(all_users)}
    mdict = {m: i for i, m in enumerate(all_movies)}

    # Инициализируем матрицы рейтингов
    train_r = np.zeros((n_m, n_u), dtype='float32')
    test_r = np.zeros((n_m, n_u), dtype='float32')

    for row in data_train:
        u_id, m_id, r = row[:3]
        train_r[mdict[m_id], udict[u_id]] = r

    for row in data_test:
        u_id, m_id, r = row[:3]
        test_r[mdict[m_id], udict[u_id]] = r

    # Маски для ненулевых рейтингов
    train_m = np.greater(train_r, 1e-12).astype('float32')
    test_m = np.greater(test_r, 1e-12).astype('float32')

    print('data matrix loaded')
    print('num of users: {}'.format(n_u))
    print('num of movies: {}'.format(n_m))
    print('num of training ratings: {}'.format(len(data_train)))
    print('num of test ratings: {}'.format(len(data_test)))

    return n_m, n_u, train_r, train_m, test_r, test_m


def local_kernel(u, v):
    dist = tf.norm(u - v, ord=2, axis=2)
    hat = tf.maximum(0., 1. - dist ** 2)

    return hat


def kernel_layer(x, n_hid=n_hid, n_dim=n_dim, activation=tf.nn.sigmoid, lambda_s=lambda_s, lambda_2=lambda_2, name=''):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        W = tf.get_variable('W', [x.shape[1], n_hid])
        n_in = x.get_shape().as_list()[1]
        u = tf.get_variable('u', initializer=tf.random.truncated_normal([n_in, 1, n_dim], 0., 1e-3))
        v = tf.get_variable('v', initializer=tf.random.truncated_normal([1, n_hid, n_dim], 0., 1e-3))
        b = tf.get_variable('b', [n_hid])

    w_hat = local_kernel(u, v)

    sparse_reg = tf.contrib.layers.l2_regularizer(lambda_s)
    sparse_reg_term = tf.contrib.layers.apply_regularization(sparse_reg, [w_hat])

    l2_reg = tf.contrib.layers.l2_regularizer(lambda_2)
    l2_reg_term = tf.contrib.layers.apply_regularization(l2_reg, [W])

    W_eff = W * w_hat  # Local kernelised weight matrix
    y = tf.matmul(x, W_eff) + b
    y = activation(y)

    return y, sparse_reg_term + l2_reg_term


def global_kernel(input, gk_size, dot_scale):
    avg_pooling = tf.reduce_mean(input, axis=1)  # Item (axis=1) based average pooling
    avg_pooling = tf.reshape(avg_pooling, [1, -1])
    n_kernel = avg_pooling.shape[1].value

    conv_kernel = tf.get_variable('conv_kernel',
                                  initializer=tf.random.truncated_normal([n_kernel, gk_size ** 2], stddev=0.1))
    gk = tf.matmul(avg_pooling, conv_kernel) * dot_scale  # Scaled dot product
    gk = tf.reshape(gk, [gk_size, gk_size, 1, 1])

    return gk


def global_conv(input, W):
    input = tf.reshape(input, [1, input.shape[0], input.shape[1], 1])
    conv2d = tf.nn.relu(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME'))

    return tf.reshape(conv2d, [conv2d.shape[1], conv2d.shape[2]])


def dcg_k(score_label, k):
    dcg, i = 0., 0
    for s in score_label:
        if i < k:
            dcg += (2 ** s[1] - 1) / np.log2(2 + i)
            i += 1
    return dcg


def ndcg_k(y_hat, y, k):
    score_label = np.stack([y_hat, y], axis=1).tolist()
    score_label = sorted(score_label, key=lambda d: d[0], reverse=True)
    score_label_ = sorted(score_label, key=lambda d: d[1], reverse=True)
    norm, i = 0., 0
    for s in score_label_:
        if i < k:
            norm += (2 ** s[1] - 1) / np.log2(2 + i)
            i += 1
    dcg = dcg_k(score_label, k)
    return dcg / norm


def call_ndcg(y_hat, y):
    ndcg_sum, num = 0, 0
    y_hat, y = y_hat.T, y.T
    n_users = y.shape[0]

    for i in range(n_users):
        y_hat_i = y_hat[i][np.where(y[i])]
        y_i = y[i][np.where(y[i])]

        if y_i.shape[0] < 2:
            continue

        ndcg_sum += ndcg_k(y_hat_i, y_i, y_i.shape[0])  # user-wise calculation
        num += 1

    return ndcg_sum / num


class My_Rec_Model:
    def __init__(self, checkpoint_path, movie2idx, idx2movie,
                 dataset_path="./dataset/", n_layers=2, gk_size=3, dot_scale=0.5):
        self.movie2idx = movie2idx
        self.idx2movie = idx2movie
        self.checkpoint_path = checkpoint_path
        self.n_layers = n_layers
        self.gk_size = gk_size
        self.dot_scale = dot_scale

        # Загружаем данные
        self.n_movies, self.n_users, self.train_r, self.train_m, self.test_r, self.test_m = load_data_1m(path=dataset_path)

        tf.reset_default_graph()

        # Построение графа
        tf.reset_default_graph()

        self.R = tf.placeholder("float32", [self.n_movies, self.n_users])

        y = self.R
        self.reg_losses = None

        for i in range(self.n_layers):
            y, reg_loss = kernel_layer(y, name=str(i))
            self.reg_losses = reg_loss if self.reg_losses is None else self.reg_losses + reg_loss

        y_dash, reg_loss = kernel_layer(y, self.n_users, activation=tf.identity, name='out')
        self.reg_losses += reg_loss

        gk = global_kernel(y_dash, self.gk_size, self.dot_scale)
        y_hat = global_conv(self.R, gk)

        for i in range(self.n_layers):
            y_hat, reg_loss = kernel_layer(y_hat, name=str(i))
            self.reg_losses += reg_loss

        self.pred_f, reg_loss = kernel_layer(y_hat, self.n_users, activation=tf.identity, name='out')
        self.reg_losses += reg_loss

        # Добавляем эмбеддинги — например возьмём y_dash, это тензор (n_movies, n_users)
        self.embeddings_f = y_dash

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        print("Restoring from:", self.checkpoint_path)
        saver.restore(self.sess, self.checkpoint_path)

    def train(self):
        n_m, n_u, train_r, train_m, test_r, test_m = load_data_1m()
        R = tf.placeholder("float", [n_m, n_u])

        y = R
        reg_losses = None

        for i in range(n_layers):
            y, reg_loss = kernel_layer(y, name=str(i))
            reg_losses = reg_loss if reg_losses is None else reg_losses + reg_loss

        pred_p, reg_loss = kernel_layer(y, n_u, activation=tf.identity, name='out')
        reg_losses = reg_losses + reg_loss

        # L2 loss
        diff = train_m * (train_r - pred_p)
        sqE = tf.nn.l2_loss(diff)
        loss_p = sqE + reg_losses

        optimizer_p = tf.contrib.opt.ScipyOptimizerInterface(loss_p,
                                                             options={'disp': True, 'maxiter': iter_p, 'maxcor': 10},
                                                             method='L-BFGS-B')
        y = R
        reg_losses = None

        for i in range(n_layers):
            y, _ = kernel_layer(y, name=str(i))

        y_dash, _ = kernel_layer(y, n_u, activation=tf.identity, name='out')

        gk = global_kernel(y_dash, gk_size, dot_scale)  # Global kernel
        y_hat = global_conv(train_r, gk)  # Global kernel-based rating matrix

        for i in range(n_layers):
            y_hat, reg_loss = kernel_layer(y_hat, name=str(i))
            reg_losses = reg_loss if reg_losses is None else reg_losses + reg_loss

        pred_f, reg_loss = kernel_layer(y_hat, n_u, activation=tf.identity, name='out')
        reg_losses = reg_losses + reg_loss

        # L2 loss
        diff = train_m * (train_r - pred_f)
        sqE = tf.nn.l2_loss(diff)
        loss_f = sqE + reg_losses

        optimizer_f = tf.contrib.opt.ScipyOptimizerInterface(loss_f,
                                                             options={'disp': True, 'maxiter': iter_f, 'maxcor': 10},
                                                             method='L-BFGS-B')

        best_rmse_ep, best_mae_ep, best_ndcg_ep = 0, 0, 0
        best_rmse, best_mae, best_ndcg = float("inf"), float("inf"), 0

        time_cumulative = 0
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            saver = tf.train.Saver()
            for i in range(epoch_p):
                tic = time()
                optimizer_p.minimize(sess, feed_dict={R: train_r})
                pre = sess.run(pred_p, feed_dict={R: train_r})

                t = time() - tic
                time_cumulative += t

                error = (test_m * (np.clip(pre, 1., 5.) - test_r) ** 2).sum() / test_m.sum()  # test error
                test_rmse = np.sqrt(error)

                error_train = (train_m * (np.clip(pre, 1., 5.) - train_r) ** 2).sum() / train_m.sum()  # train error
                train_rmse = np.sqrt(error_train)

                print('.-^-._' * 12)
                print('PRE-TRAINING')
                print('Epoch:', i + 1, 'test rmse:', test_rmse, 'train rmse:', train_rmse)
                print('Time:', t, 'seconds')
                print('Time cumulative:', time_cumulative, 'seconds')
                print('.-^-._' * 12)

            for i in range(epoch_f):
                tic = time()
                optimizer_f.minimize(sess, feed_dict={R: train_r})
                pre = sess.run(pred_f, feed_dict={R: train_r})

                t = time() - tic
                time_cumulative += t

                error = (test_m * (np.clip(pre, 1., 5.) - test_r) ** 2).sum() / test_m.sum()  # test error
                test_rmse = np.sqrt(error)

                error_train = (train_m * (np.clip(pre, 1., 5.) - train_r) ** 2).sum() / train_m.sum()  # train error
                train_rmse = np.sqrt(error_train)

                test_mae = (test_m * np.abs(np.clip(pre, 1., 5.) - test_r)).sum() / test_m.sum()
                train_mae = (train_m * np.abs(np.clip(pre, 1., 5.) - train_r)).sum() / train_m.sum()

                test_ndcg = call_ndcg(np.clip(pre, 1., 5.), test_r)
                train_ndcg = call_ndcg(np.clip(pre, 1., 5.), train_r)

                if test_rmse < best_rmse:
                    best_rmse = test_rmse
                    best_rmse_ep = i + 1
                    saver.save(sess, "checkpoints/best_rmse_model.ckpt")

                if test_mae < best_mae:
                    best_mae = test_mae
                    best_mae_ep = i + 1
                    saver.save(sess, "checkpoints/best_mae_model.ckpt")

                if best_ndcg < test_ndcg:
                    best_ndcg = test_ndcg
                    best_ndcg_ep = i + 1
                    saver.save(sess, "checkpoints/best_ndcg_model.ckpt")

                print('.-^-._' * 12)
                print('FINE-TUNING')
                print('Epoch:', i + 1, 'test rmse:', test_rmse, 'test mae:', test_mae, 'test ndcg:', test_ndcg)
                print('Epoch:', i + 1, 'train rmse:', train_rmse, 'train mae:', train_mae, 'train ndcg:', train_ndcg)
                print('Time:', t, 'seconds')
                print('Time cumulative:', time_cumulative, 'seconds')
                print('.-^-._' * 12)

        print('Epoch:', best_rmse_ep, ' best rmse:', best_rmse)
        print('Epoch:', best_mae_ep, ' best mae:', best_mae)
        print('Epoch:', best_ndcg_ep, ' best ndcg:', best_ndcg)

    def evaluate(self):
        if not hasattr(self, 'sess'):
            raise RuntimeError("Сессия TensorFlow не инициализирована.")
        if not hasattr(self, 'R') or not hasattr(self, 'pred_f'):
            raise RuntimeError("Граф модели ещё не построен. Вызови сначала __init__().")

        pred = self.sess.run(self.pred_f, feed_dict={self.R: self.train_r})
        pred = np.clip(pred, 1., 5.)

        rmse = np.sqrt(((self.test_m * (pred - self.test_r)) ** 2).sum() / self.test_m.sum())
        mae = (self.test_m * np.abs(pred - self.test_r)).sum() / self.test_m.sum()
        ndcg = call_ndcg(pred, self.test_r)

        print("Model evaluation on test set:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"NDCG: {ndcg:.4f}")

        return {
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "NDCG": round(ndcg, 4)
        }

    def predict(self, movie_names: List[str], ratings: List[float], top_k: int = 20):
        if not hasattr(self, 'sess'):
            raise RuntimeError("Сессия TensorFlow не инициализирована.")
        if not hasattr(self, 'R') or not hasattr(self, 'pred_f'):
            raise RuntimeError("Граф модели ещё не построен. Вызови сначала __init__().")

        if len(movie_names) != len(ratings):
            return {"error": "movie_names and ratings must have same length"}

        # Инициализируем матрицу рейтингов для всех пользователей
        r = np.zeros((self.n_movies, self.n_users), dtype=np.float32)

        # Выбираем пользователя, для которого хотим предсказать — 0-й индекс
        user_idx = 0

        # Заполняем рейтинги пользователя
        for mname, r_value in zip(movie_names, ratings):
            if mname not in self.movie2idx:
                continue  # пропускаем неизвестные фильмы
            movie_idx = self.movie2idx[mname]
            r[movie_idx, user_idx] = r_value

        # Получаем предсказания для всех фильмов и пользователей
        preds = self.sess.run(self.pred_f, feed_dict={self.R: r})

        # Берём предсказания только для нужного пользователя
        preds_for_user = preds[:, user_idx]

        # Исключаем уже оценённые фильмы (уже есть рейтинг)
        preds_for_user[r[:, user_idx] > 0] = -np.inf

        # Берём индексы топ фильмов
        top_idx = np.argsort(preds_for_user)[::-1][:top_k]

        # В зависимости от типа ключей idx2movie — приводим индекс к нужному виду
        sample_key = next(iter(self.idx2movie.keys()))
        if isinstance(sample_key, int):
            top_movies = [self.idx2movie[i] for i in top_idx]
        else:
            top_movies = [self.idx2movie[str(i)] for i in top_idx]

        top_ratings = [float(preds_for_user[i]) for i in top_idx]

        return {"movies": top_movies, "ratings": top_ratings}

    #
    # def warmup(self):
    #     if self.model is not None:
    #         return
    #     with open("./model/model.pkl", "rb") as f:
    #         self.model = pickle.load(f)
    #     logging.info("Model loaded from disk.")
    #
    def similar(self, movie_name: str, top_k: int = 20):
        if movie_name not in self.movie2idx:
            return {"error": f"Movie '{movie_name}' not found in index"}

        embeddings = self.sess.run(self.embeddings_f, feed_dict={self.R: self.train_r})

        movie_idx = self.movie2idx[movie_name]
        movie_emb = embeddings[movie_idx].reshape(1, -1)

        # numpy-реализация косинусного сходства
        norm_movie = movie_emb / np.linalg.norm(movie_emb, axis=1, keepdims=True)
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarities = np.dot(norm_movie, norm_embeddings.T).flatten()

        similarities[movie_idx] = -np.inf

        top_indices = similarities.argsort()[::-1][:top_k]

        sample_key = next(iter(self.idx2movie.keys()))
        if isinstance(sample_key, int):
            top_movies = [self.idx2movie[i] for i in top_indices]
        else:
            top_movies = [self.idx2movie[str(i)] for i in top_indices]

        result = [{"movie_name": name} for name in top_movies]
        return result

