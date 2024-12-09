import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.debug.metrics as met
import torch_xla.debug.profiler as xp

# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten
        return self.fc(x)

# Define dataset and dataloader
def get_data():
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='mnist_data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='mnist_data', train=False, transform=transform, download=True)
    return train_dataset, test_dataset

def train_loop(loader, model, optimizer, device):
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        xm.optimizer_step(optimizer)
        # print(f"Loss: {loss.item()}")

# Main function
def run_tpu():
    device = xm.xla_device()  # Set the TPU device
    train_dataset, test_dataset = get_data()
    
    # Create a DataLoader wrapped in ParallelLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    train_device_loader = pl.MpDeviceLoader(train_loader, device)

    # Initialize model, optimizer, and send model to TPU
    model = SimpleModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Training loop
    train_loop(train_device_loader, model, optimizer, device)
    
    # Print metrics
    # print(met.metrics_report())