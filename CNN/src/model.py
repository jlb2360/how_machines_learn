import os
import torch
from torch import nn

# set the gpu device if available
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        self.image_size = 28*28
        self.nneurons = 512
        self.noutputs = 10

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.image_size, self.nneurons), # first layer takes in image
            nn.ReLU(), # pass data through the ReLU activation function
            nn.Linear(self.nneurons, self.nneurons), # pass output through a hidden layer
            nn.ReLU(), # pass data through the ReLU activation function
            nn.Linear(self.nneurons, self.noutputs) # pass output through the output layer
        )

    def forward(self, x):
        x = self.flatten(x) # flatten to a vector
        logits = self.linear_relu_stack(x) # push vector through the network layers
        return logits
    

    def train_loop(self, dataloader, loss_fn, optimizer, batch_size):
        size = len(dataloader.dataset)

        self.train()
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            pred = self(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * batch_size + len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    def test_loop(self, dataloader, loss_fn):
        self.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(device)
                y = y.to(device)
                pred = self(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):0.1f}%, Avg loss: {test_loss:>8f} \n")

    def train_model(self, train_dataloader, test_dataloader, loss_fn, optimizer, epochs, batch_size):
        for t in range(epochs):
            print(f"Epoch {t+1}\n--------------------------------------------")
            self.train_loop(train_dataloader, loss_fn, optimizer, batch_size)
            self.test_loop(test_dataloader, loss_fn)
        print("Done!")