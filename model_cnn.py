from torch import nn
import torch.optim as optim
import lightning as L
from data_functions import get_loaders_weights_and_occurrences, DEVICE
import pickle

BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 8

train_loader, val_loader, weights, _ = get_loaders_weights_and_occurrences(BATCH_SIZE)


class MyNeuralNetwork(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 26),  # 26 outputs
        )

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv_layers(x)
        logits = self.fc_layers(x)
        return logits


def train():
    model = MyNeuralNetwork().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(NUM_EPOCHS):
        for batch_data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    filename = 'models/new_model.sav'
    pickle.dump(model, open(filename, 'wb'))
