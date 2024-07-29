import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F
import random
import itertools

# ====  Global Vars ====

DATA_PATH = "./data/"
DATA_CLASSES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
DATA_SETS = ['train', 'val', 'test']
IDX_TO_CHAR = {0: '', 1: 'e', 2: 'f', 3: 'g', 4: 'h', 5: 'i', 6: 'n', 7: 'o', 8: 'r', 9: 's', 10: 't', 11: 'u', 12: 'v', 13: 'w', 14: 'x', 15: 'z'}
CHAR_TO_IDX = {'': 0, 'e': 1, 'f': 2, 'g': 3, 'h': 4, 'i': 5, 'n': 6, 'o': 7, 'r': 8, 's': 9, 't': 10, 'u': 11, 'v': 12, 'w': 13, 'x': 14, 'z': 15}
random.seed(18)

# ====  Global Function ====

def convert_label_to_char_sequence(label):
    digit_name = DATA_CLASSES[label]
    return [CHAR_TO_IDX[char] for char in digit_name]

# ====  Data    ====
def load_data(n_train=-1, n_val=-1, n_test=-1, data_path=DATA_PATH, sample_rate=16000, n_mfcc=13):

    data = {set_name: [] for set_name in DATA_SETS}
    n_samples = {'train': n_train, 'val':n_val, 'test':n_test}

    for set_name in DATA_SETS:
        for label, category in enumerate(DATA_CLASSES):
            category_path = os.path.join(data_path, set_name, category)
            for i, file_name in enumerate(os.listdir(category_path)):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(category_path, file_name)
                    audio, _ = librosa.load(file_path, sr=sample_rate)
                    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
                    data[set_name].append((mfcc.T, label))

        random.shuffle(data[set_name])
        data[set_name] = data[set_name] if n_samples[set_name] == -1 else data[set_name][:n_samples[set_name]]

    return data

# ====  Models  ====

class LinearModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=-1)  # Apply log_softmax
        return x.unsqueeze(0)  # Add time dimension (1, batch_size, num_classes)

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
        x = F.log_softmax(x, dim=-1)  # Apply log_softmax
        return x.unsqueeze(0)  # Add time dimension (1, batch_size, num_classes)

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=-1)  # Apply log_softmax
        return x  # Shape: (batch_size, sequence_length, num_classes)

# === Training ===

def train_model(model, train_loader, val_loader, optimizer, num_epochs, input_size):
    best_accuracy = 0
    criterion = nn.CTCLoss()
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            if len(outputs.shape) == 3:
                outputs = outputs.transpose(0, 1)  # Change to (sequence_length, batch_size, num_classes) if needed
            output_lengths = torch.full(size=(outputs.shape[1],), fill_value=outputs.shape[0], dtype=torch.long)
            target_lengths = torch.LongTensor([len(t) for t in targets])
            targets = torch.cat(targets)
            loss = criterion(outputs, targets, output_lengths, target_lengths)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                decoded_outputs = torch.argmax(outputs, dim=-1)
                predicted_labels = [''.join([IDX_TO_CHAR[idx.item()] for idx in output if idx != 0]) for output in decoded_outputs]
                correct += sum([pred == DATA_CLASSES[targets[i][0].item()] for i, pred in enumerate(predicted_labels)])
                total += len(targets)

        accuracy = 100 * correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy:.2f}%')

    return best_accuracy


def concatenate_adjacent_features(data, window_size):
    result = {set_name: [] for set_name in DATA_SETS}
    for set_name in DATA_SETS:
        for sample in data[set_name]:
            input_sample = sample[0]
            if input_sample.shape[0] % window_size != 0:
                input_sample = np.vstack((input_sample, np.repeat(np.expand_dims(input_sample[-1, :], axis=0),
                                                    (window_size - input_sample.shape[0] % window_size), axis=0)))
            list_input = [np.concatenate(input_sample[i:i + window_size, :], axis=0)
                          for i in range(0, input_sample.shape[0], window_size)]
            concatenate_input = np.vstack(list_input)
            result[set_name].append((concatenate_input, sample[1]))
    return result

# ====  Experiment Code ====

n_train, n_val, n_test = 10, 10, 10
data = load_data(n_train, n_val, n_test)

print("(#) Train data samples: ", len(data["train"]))
print("(#) Validation data samples: ", len(data["val"]))
print("(#) Test data samples: ", len(data["test"]))

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

num_classes = len(IDX_TO_CHAR)
num_epochs = 10
batch_size = 32

best_config = None
best_accuracy = 0

for optimizer_name, optimizer_class, optimizer_params in optimizers:
    for model_name, model_class in models:
        for window_size in feature_windows:

            # Preprocess data with feature concatenation
            data = concatenate_adjacent_features(data, window_size)

            # Initialize model
            model = model_class(data["train"][0][0].shape[1], hidden_size, num_classes)

            # Initialize optimizer and loss function
            optimizer = optimizer_class(model.parameters(), **optimizer_params)

            # Print model and training parameters
            print(f"\nModel: {model_name}")
            print(f"Optimizer: {optimizer_name}")
            print(f"Feature window size: {window_size}")
            print(model)
            print(f"Optimizer parameters: {optimizer_params}")

            # Train and evaluate model
            accuracy = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, input_size)

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