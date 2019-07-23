#- *-coding:utf8-*-

import sys
import numpy as np
import operator
import tensorflow as tf
import read as read
from alps.framework.task.tf_task import EstimatorTask

def lfm_train(train_data, F, alpha, beta, step):
    user_vec = {}
    item_vec = {}
    for i in range(step):
        for data_slice in train_data:
            userID, movieID, rating = data_slice
            if userID not in user_vec:
                user_vec[userID] = np.random.randn(F)
            if movieID not in item_vec:
                item_vec[movieID] = np.random.randn(F)
            delta = rating - predict_LFM(user_vec[userID], item_vec[movieID])
            for dim in range(F):
                user_vec[userID][dim] += beta *(delta*item_vec[movieID][dim] - alpha*user_vec[userID][dim])
                item_vec[movieID][dim] += beta * (delta * user_vec[userID][dim] - alpha * item_vec[movieID][dim])
        beta *= 0.9
        print("Training finish " + str(i+1) + " steps.")
    return user_vec, item_vec

def predict_LFM(user_vec, item_vec):
    res = np.dot(user_vec, item_vec) / (np.linalg.norm(user_vec) * np.linalg.norm(item_vec))
    return res


def model_train_process():
    train_data = read.get_train_data(r"ratings.txt")
    user_vec, item_vec = lfm_train(train_data, 64, 0.01, 0.1, 64)
    return user_vec, item_vec

def give_recom_result(user_vec, item_vec, userID):
    recom_list = []
    if userID not in user_vec:
        return recom_list
    else:
        record = {}
        for itemID in item_vec:
            record[itemID] = predict_LFM(user_vec[userID], item_vec[itemID])
        for slice in sorted(record.items(), key = operator.itemgetter(1), reverse=True)[:10]:
            itemID = slice[0]
            score = round(slice[1], 3)
            recom_list.append([itemID, score])
        return recom_list


def display_recom_movies(userID):
    user_vec, item_vec = model_train_process()
    recom_list = give_recom_result(user_vec, item_vec, userID)
    train_data = read.get_train_data(r"ratings.txt")
    print("\n#############################################################\n")
    print("User " + str(userID) + " likes:")
    movie_info = read.get_item_info(r"movies.txt")
    for slice in train_data:
        if slice[0] == userID and slice[2] == 1:
            print(movie_info[slice[1]])
    print("\n#############################################################\n")
    print("The system recommendation List:")
    for slice in recom_list:
        recom_movie_ID = slice[0]
        print(movie_info[recom_movie_ID])


class RecommenderTask(EstimatorTask):
    def run(self):
        display_recom_movies('24')