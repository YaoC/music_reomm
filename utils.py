import pickle
import csv

DATA_PATH = "./data/"


def load_params():
    # 加载预训练的参数
    model = pickle.load('./params/dbn_parameters.pkl')
    input_size = model['input_size']
    rbm_hidden_sizes = model['rbm_hidden_sizes']
    params = model['params']
    return input_size, rbm_hidden_sizes, params


def read_csv(path):
    data_idx = {}
    data = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        idx = 0
        for l in reader:
            data.append(l[0])
            data_idx[l[0]] = idx
            idx += 1
    return data, data_idx


def load_data():
    # 1.加载用户id
    users, users_idx = read_csv('./data/users_idx.csv')
    songs, songs_idx = read_csv('./data/songs_idx.csv')




