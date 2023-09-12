import torch
import torch.optim as optim
from torchvision import datasets, transforms
from diffusion_model import DiffusionModel
import yaml
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file (default: config.yaml)")
    return parser.parse_args()

def train_and_evaluate(config):
    # Hyperparameters from config
    lr = config["lr"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Data loaders
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root=config["data_root"], train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root=config["data_root"], train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer
    model = DiffusionModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # TensorBoard
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

    def test_model():
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                outputs = model(data)
                loss = criterion(outputs, data.view(data.size(0), -1))
                test_loss += loss.item() * data.size(0)  # Multiply by batch size
        return test_loss / len(test_loader.dataset)

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, _) in enumerate(train_loader):  
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data.view(data.size(0), -1))
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}/{epochs} Batch {batch_idx}/{len(train_loader)} Training Loss: {loss.item()}")
                writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)

        # Compute test loss at the end of the epoch
        test_loss = test_model()
        print(f"Epoch {epoch}/{epochs} Test Loss: {test_loss}")
        writer.add_scalar('Test Loss', test_loss, epoch)

    writer.close()
    torch.save(model.state_dict(), config["model_save_path"])
    
    return test_loss

if __name__ == "__main__":
    args = parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    train_and_evaluate(config)
