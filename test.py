#!/home/pthinh/tmp/torch_env/bin/python3
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import matplotlib.pyplot as plt



def read_idx_images(path):
    with open(path, "rb") as f:
        magic, n, rows, cols = np.frombuffer(f.read(16), dtype=">i4")
        assert magic == 2051, "Not image file"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols).copy()
def read_idx_labels(path):
    with open(path, "rb") as f:
        magic, n = np.frombuffer(f.read(8), dtype=">i4")
        assert magic == 2049, "Not label file"
        return np.frombuffer(f.read(), dtype=np.uint8).copy()

class IdxDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images = read_idx_images(images_path)
        self.labels = read_idx_labels(labels_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


device = "cuda" if torch.cuda.is_available() else "cpu"

transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True)])
test_data = IdxDataset(
    "/home/pthinh/tmp/test_torch/data_train/t10k-images.idx3-ubyte",
    "/home/pthinh/tmp/test_torch/data_train/t10k-labels.idx1-ubyte",
    transform=transform
)
train_data = IdxDataset(
    "/home/pthinh/tmp/test_torch/data_train/train-images.idx3-ubyte",
    "/home/pthinh/tmp/test_torch/data_train/train-labels.idx1-ubyte",
    transform=transform
)
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)


# img, label = test_data[0]
# print(img.shape, img.dtype, label)   
# print(img.min(), img.max())           

# train_features, train_labels = next(iter((train_dataloader)))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img1, label = train_features[0].squeeze(), train_labels[0].item()

# plt.imshow(img.squeeze(), cmap="gray")
# plt.title(label)
# plt.savefig("test.png")

# plt.figure()
# plt.imshow(img1, cmap="gray")
# plt.title(label)
# plt.savefig("test2.png")

learning_rate = 1e-3
batch_size = 64
epochs = 5
model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
for t in range(epochs):
    print(f"Epoch {t+1}\n--------------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
# torch.save(model.state_dict(), "model_regconize_number.pth")

