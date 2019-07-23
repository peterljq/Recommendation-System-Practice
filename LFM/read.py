#- *-coding:utf8-*-
import os
def get_item_info(input_path):
    if not os.path.exists(input_path):
        return {}
    item_info = {}
    line_num = 0
    fp = open(input_path)
    for line in fp:
        if line_num == 0:
            line_num += 1
            continue
        items = line.split(',')
        if len(items) < 3:
            continue
        elif len(items) == 3:
            movieID = items[0]
            title = items[1]
            genre = items[2]
        else:
            movieID = items[0]
            genre = items[-1]
            title = ','.join(items[1:-1])
        item_info[movieID] = [title, genre]
    fp.close()
    return item_info


def get_avg_score(input_path):
    if not os.path.exists(input_path):
        return {}
    line_num = 0
    record_dict = {}
    score_dict = {}
    fp = open(input_path)
    for line in fp:
        if (line_num == 0):
            line_num += 1
            continue
        items = line.split(',')
        if len(items) <= 3:
            continue
        userID = items[0]
        movieID = items[1]
        rating = float(items[2])
        timeStamp = items[3]
        if movieID not in record_dict:
            record_dict[movieID] = [0, 0]
        record_dict[movieID][0] += 1
        record_dict[movieID][1] += rating
    fp.close()
    for movieID in record_dict:
        score_dict[movieID] = round(record_dict[movieID][1]/record_dict[movieID][0], 3)
    return score_dict

def get_train_data(input_path):
    if not os.path.exists(input_path):
        print("TRAIN DATA PATH NOT EXISTS.")
        return {}
    score_dict = get_avg_score(input_path)
    pos_dict = {}
    neg_dict = {}
    train_data = []
    line_num = 0
    fp = open(input_path)
    for line in fp:
        if (line_num == 0):
            line_num += 1
            continue
        items = line.split(',')
        if len(items) <= 3:
            continue
        userID, movieID, rating = items[0], items[1], float(items[2])
        if userID not in pos_dict:
            pos_dict[userID] = []
        if userID not in neg_dict:
            neg_dict[userID] = []
        if rating >= 4.0:
            pos_dict[userID].append([movieID, 1])
        else:
            avg_score = score_dict[movieID]
            neg_dict[userID].append([movieID, avg_score])
    fp.close()
    for userID in pos_dict:
        data_num = min(len(pos_dict[userID]), len(neg_dict[userID]))
        if data_num <= 0:
            continue
        else:
            data_count = 0
            for rec in pos_dict[userID]:
                if data_count < data_num:
                    train_data.append([userID, rec[0], rec[1]])
                    data_count += 1
                else:
                    continue
            sorted_neg_list = sorted(neg_dict[userID], key = lambda element:element[1], reverse = True)[0:data_num]
            data_count = 0
            for rec in sorted_neg_list:
                if data_count < data_num:
                    train_data.append([userID, rec[0], 0])
                    data_count += 1
                else:
                    continue
            data_count = 0
    return train_data