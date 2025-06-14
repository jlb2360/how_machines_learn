from torch import nn
import torch

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        start_channels = 1 # grey picture
        out_channel1 = 32 # number of filters for first conv
        out_channel2 = 64 # number of filters for second channel
        final_image_size = 7*7 # final image size after two pools on 28x28 image
        output_size = 10 # number of digits

        self.cnn_stack = nn.Sequential(
            nn.Conv2d(in_channels=start_channels, out_channels=out_channel1, kernel_size=3, stride=1, padding=1), # first convolution
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # end of first network
            nn.Conv2d(in_channels=out_channel1, out_channels=out_channel2, kernel_size=3, stride=1, padding=1), # secend convolution
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(), # flatten the output
            nn.Linear(out_channel2*final_image_size, output_size) # 7x7 image after last convolutions which output digits
        )

    def forward(self, x):
        logits = self.cnn_stack(x) # push vector through the network layers
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