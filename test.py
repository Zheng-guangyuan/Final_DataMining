import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import OrderedDict


def load_data(file_path):
    data = pd.read_table(file_path, sep='\t', header=None)
    data = data.iloc[:, :3].values
    return data


def build_rating_matrix(data):
    num_users = max([d[0] for d in data])
    num_items = max([d[1] for d in data])
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


class AutoEncoder(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, sparse_reg=1e-5):
        super(AutoEncoder, self).__init__()

        # Encoder
        encoder_layers = OrderedDict()
        for i in range(len(hidden_size) - 1):
            encoder_layers[f'enc_linear{i}'] = nn.Linear(hidden_size[i], hidden_size[i + 1])
            encoder_layers[f'enc_drop{i}'] = nn.Dropout(dropout)
            encoder_layers[f'enc_relu{i}'] = nn.ReLU()
        self.encoder = nn.Sequential(encoder_layers)

        # Decoder
        decoder_layers = OrderedDict()
        for i in range(len(hidden_size) - 1, 0, -1):
            decoder_layers[f'dec_linear{i}'] = nn.Linear(hidden_size[i], hidden_size[i - 1])
            decoder_layers[f'dec_relu{i}'] = nn.Sigmoid()
        self.decoder = nn.Sequential(decoder_layers)

        # L1 regularization weight
        self.sparse_reg = sparse_reg

    def forward(self, x):
        # Normalize input to the range [0, 1]
        x = (x - 1) / 4.0

        # Pass through encoder
        encoded = self.encoder(x)

        # Pass through decoder
        decoded = self.decoder(encoded)

        # Rescale to original range [1, 5]
        decoded = decoded * 4.0 + 1

        return decoded, encoded

    def l1_penalty(self, encoded):
        # L1 penalty for sparse regularization
        return self.sparse_reg * torch.sum(torch.abs(encoded))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# parameters
dropout = 0.1
sparse_reg = 1e-5
batch_size = 128
num_epochs = 25
learning_rate = 0.001
user_based = False

if __name__ == '__main__':
    start = datetime.now()
    train_data = load_data('ml-100k/u2.base')
    test_data = load_data('ml-100k/u2.test')

    train_matrix, train_mask, num_users, num_items = build_rating_matrix(train_data)
    test_matrix, test_mask = build_test_matrix(test_data, num_users, num_items)

    # Prepare dataset and dataloader
    trainset = RatingDataset(train_matrix, train_mask, num_users, num_items, user_based)
    testset = RatingDataset(test_matrix, test_mask, num_users, num_items, user_based)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    if user_based:
        input_size = num_items
    else:
        input_size = num_users

    hidden_size = [input_size, 256, 128, 64]

    # Initialize model, loss function, and optimizer
    model = AutoEncoder(hidden_size, dropout, sparse_reg).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data, mask in train_loader:
            data, mask = data.to(device), mask.to(device)
            optimizer.zero_grad()
            output, encoded = model(data)
            loss = criterion(output * mask, data * mask) + model.l1_penalty(encoded)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        train_loss /= len(trainset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        train_matrix_tensor, train_mask_tensor = trainset.get_mat()
        test_matrix_tensor, test_mask_tensor = testset.get_mat()

        train_matrix_tensor = train_matrix_tensor.to(device)
        train_mask_tensor = train_mask_tensor.to(device)
        test_matrix_tensor = test_matrix_tensor.to(device)
        test_mask_tensor = test_mask_tensor.to(device)

        predicted_train, _ = model(train_matrix_tensor)

        train_loss = criterion(predicted_train * train_mask_tensor, train_matrix_tensor * train_mask_tensor)
        test_loss = criterion(predicted_train * test_mask_tensor, test_matrix_tensor * test_mask_tensor)

        print(f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    end = datetime.now()
    print("Total time: %s" % str(end - start))
