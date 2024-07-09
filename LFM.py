import pandas as pd
import math
import numpy as np
from matplotlib import pyplot as plt


# load data from trian / test dataset
def load_data(file_path):
    data = pd.read_table(file_path, sep='\t', header=None)
    data = data.iloc[:, :3].values
    return data


# Initial users' and items' latent factor vectors
def initialize_factors(num_users, num_items, num_factors):
    user_factors = np.random.randn(num_users, num_factors)
    item_factors = np.random.randn(num_items, num_factors)
    b_item = np.random.randn(num_items)
    b_user = np.random.randn(num_users)

    return user_factors, item_factors, b_user, b_item


# train Latent factor model
def train_model(train_data, num_users, num_items, num_factors, mean_arr, learning_rate, reg_param, epochs):
    user_factors, item_factors, b_user, b_item = initialize_factors(num_users, num_items, num_factors)
    loss_history = []  # Record every epoch's loss
    for epoch in range(epochs):
        total_MSE = 0
        index = np.random.permutation(len(train_data))
        for idx in index:
            user, item, rating = train_data[idx]
            user -= 1  # adjust index start from 0
            item -= 1
            prediction = mean_arr + b_user[user] + b_item[item] + np.dot(user_factors[user], item_factors[item].T)
            error = rating - prediction
            total_MSE += error ** 2

            # Update bias and latent factor vectors
            b_user[user] += learning_rate * (error - reg_param * b_user[user])
            b_item[item] += learning_rate * (error - reg_param * b_item[item])
            temp = user_factors[user]
            user_factors[user] += learning_rate * (error * item_factors[item] - reg_param * user_factors[user])
            item_factors[item] += learning_rate * (error * temp - reg_param * item_factors[item])

        avg_MSE = total_MSE / len(train_data)
        RMSE = math.sqrt(avg_MSE)
        loss_history.append(RMSE)
        print("epoch:", epoch + 1, "train_data RMSE:", RMSE)
    return user_factors, item_factors, b_user, b_item, loss_history


# predict
def predict(user_factors, item_factors, b_user, b_item, mean_arr, user, item):
    return mean_arr + b_user[user-1] + b_item[item-1] + np.dot(user_factors[user-1], item_factors[item-1].T)


# Calculate RMSE
def calculate_rmse(test_data, user_factors, item_factors, b_user, b_item, mean_arr):
    mse = 0
    for user, item, rating in test_data:
        predicted_rating = predict(user_factors, item_factors, b_user, b_item, mean_arr, user, item)
        mse += (predicted_rating - rating) ** 2
    rmse = np.sqrt(mse / len(test_data))
    return rmse


train_data = load_data('ml-100k/u1.base')
test_data = load_data('ml-100k/u1.test')

num_users = max(max(data[0] for data in train_data), max(data[0] for data in test_data))
num_items = max(max(data[1] for data in train_data), max(data[1] for data in test_data))

num_factors = 10  # Number of latent factors
mean_arr = np.mean(train_data[:, 2])

# trian Latent factor model
learning_rate = 0.01
reg_param = 0.1
epochs = 50
user_factors, item_factors, b_user, b_item, loss_history = train_model(train_data, num_users, num_items, num_factors, mean_arr, learning_rate, reg_param, epochs)

rmse = calculate_rmse(test_data, user_factors, item_factors, b_user, b_item, mean_arr)
print('RMSE:', rmse)

plt.plot(range(1, epochs + 1), loss_history)
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.show()