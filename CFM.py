import pandas as pd
from scipy.spatial.distance import cosine
import numpy as np


# load data from trian / test dataset
def load_data(file_path):
    data = pd.read_table(file_path, sep='\t', header=None)
    data = data.iloc[:, :3].values
    return data


# Create User-rating matrix
def build_rating_matrix(data):
    num_users = max([d[0] for d in data])
    num_items = max([d[1] for d in data])
    rating_matrix = np.zeros((num_users, num_items))
    for user_id, item_id, rating in data:
        rating_matrix[user_id - 1, item_id - 1] = rating
    return rating_matrix


# Calculate the similarity between users (cosine similarity)
def cosine_similarity(user_matrix):
    num_users, num_items = user_matrix.shape
    sim_matrix = np.zeros((num_users, num_users))
    for i in range(num_users):
        for j in range(i, num_users):  # only calculate upper triangular part
            sim = 1 - cosine(user_matrix[i], user_matrix[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim  
    return sim_matrix


# User-based CFM
def predict_user_based(user_matrix, sim_matrix, k=5):
    num_users, num_items = user_matrix.shape
    pred_matrix = np.zeros((num_users, num_items))
    for i in range(num_users):
        similar_users = np.argsort(sim_matrix[i])[::-1][:k]  # Get the k users that are most similar
        for j in range(num_items):
            if user_matrix[i, j] == 0:  # Unrated items
                pred_rating = 0
                sim_sum = 0
                for user in similar_users:
                    if user_matrix[user, j] != 0:  # rated items by similar users
                        pred_rating += sim_matrix[i, user] * user_matrix[user, j]
                        sim_sum += sim_matrix[i, user]
                if sim_sum != 0:
                    pred_matrix[i, j] = pred_rating / sim_sum
    return pred_matrix


# Calculate RMSE
def rmse(actual_matrix, pred_matrix):
    non_zero_indices = np.nonzero(actual_matrix)
    num_ratings = len(non_zero_indices[0])
    error = 0
    for i in range(num_ratings):
        user = non_zero_indices[0][i]
        item = non_zero_indices[1][i]
        error += (actual_matrix[user, item] - pred_matrix[user, item]) ** 2
    return np.sqrt(error / num_ratings)
    

train_data = load_data('ml-100k/u1.base')
test_data = load_data('ml-100k/u1.test')

train_matrix = build_rating_matrix(train_data)
test_matrix = build_rating_matrix(test_data)

user_similarity = cosine_similarity(train_matrix)
user_based_pred = predict_user_based(train_matrix, user_similarity)

user_based_rmse = rmse(test_matrix, user_based_pred)
print('User-based CF RMSE:', user_based_rmse)
