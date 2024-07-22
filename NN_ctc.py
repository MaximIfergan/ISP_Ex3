import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def load_data(data_path="data", sample_rate=16000, n_mfcc=13):
    categories = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    sets = ['train', 'val', 'test']
    data = {set_name: [] for set_name in sets}
    labels = {set_name: [] for set_name in sets}

    for set_name in sets:
        for label, category in enumerate(categories):
            category_path = os.path.join(data_path, set_name, category)
            for file_name in os.listdir(category_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(category_path, file_name)
                    audio, _ = librosa.load(file_path, sr=sample_rate)
                    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
                    data[set_name].append(mfcc.T)  # Transpose to have time as the first dimension
                    labels[set_name].append(label)

    return (np.array(data['train']), np.array(labels['train']),
            np.array(data['val']), np.array(labels['val']),
            np.array(data['test']), np.array(labels['test']))

def preprocess_data(train_data, val_data, test_data):
    # Find the maximum length of sequences
    max_length = max(max(len(seq) for seq in train_data),
                     max(len(seq) for seq in val_data),
                     max(len(seq) for seq in test_data))

    # Pad sequences to have the same length
    def pad_sequences(data):
        return np.array([np.pad(seq, ((0, max_length - len(seq)), (0, 0)), mode='constant') for seq in data])

    train_data_padded = pad_sequences(train_data)
    val_data_padded = pad_sequences(val_data)
    test_data_padded = pad_sequences(test_data)

    # Reshape data to 2D (batch_size, time_steps * features)
    train_data_reshaped = train_data_padded.reshape(train_data_padded.shape[0], -1)
    val_data_reshaped = val_data_padded.reshape(val_data_padded.shape[0], -1)
    test_data_reshaped = test_data_padded.reshape(test_data_padded.shape[0], -1)

    # Normalize data
    mean = np.mean(train_data_reshaped)
    std = np.std(train_data_reshaped)
    train_data_normalized = (train_data_reshaped - mean) / std
    val_data_normalized = (val_data_reshaped - mean) / std
    test_data_normalized = (test_data_reshaped - mean) / std

    return train_data_normalized, val_data_normalized, test_data_normalized

# Usage example:
train_data, train_labels, val_data, val_labels, test_data, test_labels = load_data()
train_data, val_data, test_data = preprocess_data(train_data, val_data, test_data)

print("Train data shape:", train_data.shape)
print("Validation data shape:", val_data.shape)
print("Test data shape:", test_data.shape)

class LinearModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ConvModel, self).__init__()
        self.conv = nn.Conv1d(1, hidden_size, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(hidden_size * input_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.relu(self.conv(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])  # Use the last hidden state
        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = 100 * correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy:.2f}%')

    return best_accuracy


def concatenate_features(data, window_size):
    result = []
    for sample in data:
        concatenated = np.concatenate([sample[i:i + window_size] for i in range(len(sample) - window_size + 1)], axis=1)
        result.append(concatenated)
    return np.array(result)


# Load and preprocess data
train_data, train_labels, val_data, val_labels, test_data, test_labels = load_data()

# Define hyperparameters and options
optimizers = [
    ('SGD', optim.SGD, {'lr': 0.01, 'momentum': 0.9}),
    ('Adam', optim.Adam, {'lr': 0.001}),
    ('AdamW', optim.AdamW, {'lr': 0.001})
]

models = [
    ('Linear', LinearModel),
    ('Conv', ConvModel),
    ('RNN', RNNModel)
]

feature_windows = [1, 3, 5]  # 1 means no concatenation

hidden_size = 128
num_classes = 10
num_epochs = 30
batch_size = 32

best_config = None
best_accuracy = 0

for optimizer_name, optimizer_class, optimizer_params in optimizers:
    for model_name, model_class in models:
        for window_size in feature_windows:
            # Preprocess data with feature concatenation
            train_data_concat = concatenate_features(train_data, window_size)
            val_data_concat = concatenate_features(val_data, window_size)

            train_data_processed, val_data_processed, _ = preprocess_data(train_data_concat, val_data_concat,
                                                                          val_data_concat)

            input_size = train_data_processed.shape[1]

            # Create datasets and dataloaders
            train_dataset = TensorDataset(torch.FloatTensor(train_data_processed), torch.LongTensor(train_labels))
            val_dataset = TensorDataset(torch.FloatTensor(val_data_processed), torch.LongTensor(val_labels))

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Initialize model
            model = model_class(input_size, hidden_size, num_classes)

            # Initialize optimizer and loss function
            optimizer = optimizer_class(model.parameters(), **optimizer_params)
            criterion = nn.CrossEntropyLoss()

            # Print model and training parameters
            print(f"\nModel: {model_name}")
            print(f"Optimizer: {optimizer_name}")
            print(f"Feature window size: {window_size}")
            print(f"Input size: {input_size}")
            print(model)
            print(f"Optimizer parameters: {optimizer_params}")

            # Train and evaluate model
            accuracy = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

            # Update best configuration if necessary
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = {
                    'model': model_name,
                    'optimizer': optimizer_name,
                    'feature_window': window_size,
                    'accuracy': accuracy
                }

print("\nBest configuration:")
print(f"Model: {best_config['model']}")
print(f"Optimizer: {best_config['optimizer']}")
print(f"Feature window size: {best_config['feature_window']}")
print(f"Best validation accuracy: {best_config['accuracy']:.2f}%")