import torch
import torch.nn as nn
import torch.optim as optim
from network import AutoEncoder
from datetime import datetime
from datapreprocessing import load_data,  build_rating_matrix, build_test_matrix, RatingDataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# parameters
dropout = 0.1
sparse_reg = 1e-5
batch_size = 64
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
    print("Total time: %s" % str(end-start))
