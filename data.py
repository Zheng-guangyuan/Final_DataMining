import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_data(file_path):
    data = pd.read_table(file_path, sep='\t', header=None)
    data = data.iloc[:, :3].values
    return data


def build_rating_matrix(data):
    num_users = int(max([d[0] for d in data]))
    num_items = int(max([d[1] for d in data]))
    print(num_users, num_items)
    rating_matrix = np.zeros((num_users, num_items))
    mask = np.zeros((num_users, num_items))
    for user_id, item_id, rating in data:
        rating_matrix[user_id - 1, item_id - 1] = rating
        mask[user_id - 1, item_id - 1] = 1
    return rating_matrix, mask, num_users, num_items


def build_test_matrix(data, num_users, num_items):
    test_matrix = np.zeros((num_users, num_items))
    mask = np.zeros((num_users, num_items))
    for user_id, item_id, rating in data:
        test_matrix[user_id - 1, item_id - 1] = rating
        mask[user_id - 1, item_id - 1] = 1
    return test_matrix, mask


class RatingDataset(Dataset):
    def __init__(self, rating_matrix, mask, num_users, num_items, user_based=True):
        self.mat = torch.from_numpy(rating_matrix).float()
        self.mask = torch.from_numpy(mask).float()
        self.mode = user_based
        self.num_users = num_users
        self.num_items = num_items

        # item-based(transpose the matrix)
        if not self.mode:
            self.mat = self.mat.t()
            self.mask = self.mask.t()

    def __getitem__(self, index):
        return self.mat[index], self.mask[index]

    def __len__(self):
        if self.mode:
            return self.num_users
        return self.num_items

    def get_mat(self):
        return self.mat, self.mask
