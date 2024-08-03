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
IDX_TO_CHAR = {0: '', 1: 'e', 2: 'f', 3: 'g', 4: 'h', 5: 'i', 6: 'n', 7: 'o', 8: 'r', 9: 's', 10: 't', 11: 'u', 12: 'v',
               13: 'w', 14: 'x', 15: 'z'}
CHAR_TO_IDX = {'': 0, 'e': 1, 'f': 2, 'g': 3, 'h': 4, 'i': 5, 'n': 6, 'o': 7, 'r': 8, 's': 9, 't': 10, 'u': 11, 'v': 12,
               'w': 13, 'x': 14, 'z': 15}
MAX_TARGET_SEQ_LEN = max([len(class_label) for class_label in DATA_CLASSES])
random.seed(18)

# ====  Global Function ====

def convert_label_to_char_sequence(label):
    digit_name = DATA_CLASSES[label]
    return [CHAR_TO_IDX[char] for char in digit_name]


# ====  Data    ====
def load_data(n_train=-1, n_val=-1, n_test=-1, data_path=DATA_PATH, sample_rate=16000, n_mfcc=13):
    data = {set_name: [] for set_name in DATA_SETS}
    n_samples = {'train': n_train, 'val': n_val, 'test': n_test}

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
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x = x.transpose(1, 2)  # Change to (batch_size, input_size, seq_len)
        x = self.relu(self.conv(x))
        x = x.transpose(1, 2)  # Change back to (batch_size, seq_len, hidden_size)
        x = self.fc(x)
        x = F.log_softmax(x, dim=-1)  # Apply log_softmax
        return x  # Shape: (batch_size, seq_len, num_classes)


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

def train_model(model, data, optimizer, num_epochs, batch_size, max_seq_len):
    best_accuracy = 0
    criterion = nn.CTCLoss()
    model.train()
    for epoch in range(num_epochs):

        batch_data_input = [[sample[0] for sample in data['train'][i:i + batch_size]]
                            for i in range(0, len(data['train']), batch_size)]
        batch_data_target = [[sample[1] for sample in data['train'][i:i + batch_size]]
                             for i in range(0, len(data['train']), batch_size)]

        for inputs, targets in zip(batch_data_input, batch_data_target):
            optimizer.zero_grad()

            # Store original input lengths
            input_lengths = torch.LongTensor([x.shape[0] for x in inputs])

            # Pad inputs to max_seq_len
            padded_inputs = [torch.nn.functional.pad(torch.FloatTensor(x), (0, 0, 0, max_seq_len - x.shape[0])) for x in
                             inputs]
            padded_inputs = torch.stack(padded_inputs)

            outputs = model(padded_inputs)  # Shape: (batch_size, max_seq_len, num_classes)

            # Reshape outputs to (max_seq_len, batch_size, num_classes)
            outputs = outputs.squeeze(0).transpose(0, 1)

            # Convert targets to tensor and pad to MAX_TARGET_SEQ_LEN
            target_lengths = torch.LongTensor([len(convert_label_to_char_sequence(t)) for t in targets])
            padded_targets = [torch.nn.functional.pad(torch.LongTensor(convert_label_to_char_sequence(t)),
                                                      (0, MAX_TARGET_SEQ_LEN - len(convert_label_to_char_sequence(t))))
                              for t in targets]
            padded_targets = torch.stack(padded_targets)

            loss = criterion(outputs, padded_targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()


        accuracy = validate_model(model, data, batch_size, max_seq_len, criterion)
        print(f"accuracy: {accuracy}")

        # # Validation
        # model.eval()
        # correct = 0
        # total = 0
        #
        # # with torch.no_grad():
        # #     for inputs, targets in val_loader:
        # #         outputs = model(inputs)
        # #         decoded_outputs = torch.argmax(outputs, dim=-1)
        # #         predicted_labels = [''.join([IDX_TO_CHAR[idx.item()] for idx in output if idx != 0]) for output in decoded_outputs]
        # #         correct += sum([pred == DATA_CLASSES[targets[i][0].item()] for i, pred in enumerate(predicted_labels)])
        # #         total += len(targets)
        #
        # accuracy = 100 * correct / total
        # if accuracy > best_accuracy:
        #     best_accuracy = accuracy
        # print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy:.2f}%')

    return accuracy


def validate_model(model, data, batch_size, max_seq_len, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    batch_data_input = [[sample[0] for sample in data['val'][i:i + batch_size]]
                        for i in range(0, len(data['val']), batch_size)]
    batch_data_target = [[sample[1] for sample in data['val'][i:i + batch_size]]
                         for i in range(0, len(data['val']), batch_size)]

    with torch.no_grad():
        for inputs, targets in zip(batch_data_input, batch_data_target):
            # Store original input length
            input_lengths = torch.LongTensor([x.shape[0] for x in inputs])

            # Pad inputs to max_seq_len
            padded_inputs = [torch.nn.functional.pad(torch.FloatTensor(x), (0, 0, 0, max_seq_len - x.shape[0])) for x in inputs]
            padded_inputs = torch.stack(padded_inputs)

            outputs = model(padded_inputs)  # Shape: (batch_size, max_seq_len, num_classes)

            # Transpose outputs to (max_seq_len, batch_size, num_classes)
            outputs = outputs.squeeze(0).transpose(0, 1)

            # Convert targets to tensor and pad to MAX_TARGET_SEQ_LEN
            target_lengths = torch.LongTensor([len(convert_label_to_char_sequence(t)) for t in targets])
            padded_targets = [torch.nn.functional.pad(torch.LongTensor(convert_label_to_char_sequence(t)),
                                                      (0, MAX_TARGET_SEQ_LEN - len(convert_label_to_char_sequence(t))))
                              for t in targets]
            padded_targets = torch.stack(padded_targets)

            loss = criterion(outputs, padded_targets, input_lengths, target_lengths)
            total_loss += loss.item()

            # Decoding
            decoded_outputs = torch.argmax(outputs.transpose(0, 1), dim=-1)
            predicted_labels = [''.join([IDX_TO_CHAR[idx.item()] for idx in output if idx != 0]) for output in decoded_outputs]
            correct += sum([pred == DATA_CLASSES[targets[i]] for i, pred in enumerate(predicted_labels)])
            total += len(targets)

    accuracy = 100 * correct / total
    return accuracy


def concatenate_adjacent_features(data, window_size):
    max_seq_len = 0
    result = {set_name: [] for set_name in DATA_SETS}
    for set_name in DATA_SETS:
        for sample in data[set_name]:
            input_sample = sample[0]
            if input_sample.shape[0] % window_size != 0:
                input_sample = np.vstack((input_sample, np.repeat(np.expand_dims(input_sample[-1, :], axis=0),
                                                                  (window_size - input_sample.shape[0] % window_size),
                                                                  axis=0)))
            list_input = [np.concatenate(input_sample[i:i + window_size, :], axis=0)
                          for i in range(0, input_sample.shape[0], window_size)]
            concatenate_input = np.vstack(list_input)
            result[set_name].append((concatenate_input, sample[1]))
            max_seq_len = max(max_seq_len, concatenate_input.shape[0])
    return result, max_seq_len


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
batch_size = 8

best_config = None
best_accuracy = 0

for window_size in feature_windows:

    # Preprocess data with feature concatenation
    data, max_seq_len = concatenate_adjacent_features(data, window_size)

    for optimizer_name, optimizer_class, optimizer_params in optimizers:
        for model_name, model_class in models:

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
            accuracy = train_model(model, data, optimizer, num_epochs, batch_size, max_seq_len)

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
