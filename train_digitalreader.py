import pandas as pd
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

# === Custom Dataset ===
class DigitDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        img_path = f"{self.img_dir}/{row['filename']}"
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor([float(row['value']) / 10000.0])  # normalize
        return image, label

# === Model ===
class DigitRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1").features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, 1)

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

# === Training Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ColorJitter(0.2, 0.2),
    transforms.ToTensor()
])

dataset = DigitDataset(
    csv_file='D:/projectCPE/dataset_digital/labels.csv',
    img_dir='D:/projectCPE/dataset_digital/images_digital',
    transform=transform
)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=8)

model = DigitRegressor().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# === Train Loop ===
for epoch in range(30):
    model.train()
    total_loss = 0
    for images, labels in train_dl:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_dl):.4f}")

torch.save(model.state_dict(), "digital_reader.pt")
print(" Training complete. Model saved as digital_reader.pt")
