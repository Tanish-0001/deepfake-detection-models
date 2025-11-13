import os
import cv2
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset    
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import multiprocessing

from src.model_core import Two_Stream_Net


class CelebDFDataset(Dataset):
    """
    PyTorch Dataset for Celeb-DF-v2 deepfake video dataset.

    Directory structure:
    Celeb-DF-v2/
        ├── Celeb-real/
        ├── YouTube-real/
        ├── Celeb-synthesis/
        ├── List_of_testing_videos.txt
    """

    def __init__(self, root_dir, split="train", frame_skip=5, transform=None):
        """
        Args:
            root_dir (str): Path to 'Celeb-DF-v2' directory.
            split (str): 'train' or 'test'.
            frame_skip (int): Extract every Nth frame.
            transform: torchvision transform for data augmentation/preprocessing.
        """
        self.root_dir = root_dir
        self.split = split
        self.frame_skip = frame_skip
        self.transform = transform

        # Read testing video list (label + relative path)
        test_file = os.path.join(root_dir, "List_of_testing_videos.txt")
        test_videos = []
        if os.path.exists(test_file):
            with open(test_file, "r") as f:
                for line in f:
                    label, rel_path = line.strip().split()
                    label = int(label)
                    test_videos.append((rel_path, label))

        # Collect all video paths
        all_videos = []
        for subfolder in ["Celeb-real", "YouTube-real", "Celeb-synthesis"]:
            folder_path = os.path.join(root_dir, subfolder)
            if not os.path.exists(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if fname.endswith((".mp4", ".avi", ".mov")):
                    path = os.path.join(subfolder, fname)
                    # Label: 1 = real, 0 = fake
                    label = 1 if "synthesis" not in subfolder.lower() else 0
                    all_videos.append((path, label))

        # Split data
        test_set_paths = set([v[0] for v in test_videos])
        if split == "train":
            self.video_list = [(p, l) for (p, l) in all_videos if p not in test_set_paths]
        else:
            self.video_list = test_videos

        print(f"[CelebDFDataset] Loaded {len(self.video_list)} {split} videos.")

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        rel_path, label = self.video_list[idx]
        video_path = os.path.join(self.root_dir, rel_path)

        # Capture video frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # Randomly pick a frame (every Nth frame)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = random.randint(0, max(0, frame_count - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"Failed to read frame from {video_path}")

        # Convert BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply transform
        if self.transform:
            frame = self.transform(frame)
        else:
            transform_default = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5]),
            ])
            frame = transform_default(frame)

        return frame, torch.tensor(label, dtype=torch.long)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    preds_all, labels_all = [], []

    for frames, labels in tqdm(dataloader, desc="Training", leave=False):
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs, _, _ = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * frames.size(0)
        preds = torch.argmax(outputs, dim=1)
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(labels_all, preds_all)
    return epoch_loss, epoch_acc



@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds_all, labels_all = [], []

    for frames, labels in tqdm(dataloader, desc="Validating", leave=False):
        frames = frames.to(device)
        labels = labels.to(device)

        outputs, _, _ = model(frames)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * frames.size(0)
        preds = torch.argmax(outputs, dim=1)
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(labels_all, preds_all)

    labels_np = np.array(labels_all)
    preds_np = np.array(preds_all)
    val_real_ratio = (labels_np == 1).mean()
    pred_real_ratio = (preds_np == 1).mean()
    print(f"[VAL CHECK] label real ratio: {val_real_ratio:.4f}, pred real ratio: {pred_real_ratio:.4f}, unique preds: {np.unique(preds_np, return_counts=True)}")

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(labels_all, preds_all)
    f1_macro = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    f1_pos = f1_score(labels_all, preds_all, pos_label=1, average="binary", zero_division=0)
    print(f"[VAL METRICS] acc: {epoch_acc:.4f} | f1_macro: {f1_macro:.4f} | f1_pos(=real): {f1_pos:.4f}")

    return epoch_loss, epoch_acc


def main():
    # Dataset / dataloaders
    train_ds = CelebDFDataset(
        root_dir="./Celeb-DF-v2",
        split="train",
        transform=None
    )

    test_ds = CelebDFDataset(
        root_dir="./Celeb-DF-v2",
        split="test"
    )

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    # Device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Two_Stream_Net().to(device)

    model.load_state_dict(torch.load("./checkpoints_celeb_df/best_model.pth", weights_only=True))

    criterion = nn.CrossEntropyLoss()     # Binary classification (real=0, fake=1)
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    num_epochs = 10
    save_dir = "./checkpoints_celeb_df"
    os.makedirs(save_dir, exist_ok=True)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"✅ Saved new best model (Val Acc = {val_acc:.4f})")

        scheduler.step()

    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    # Required on Windows to safely spawn dataloader worker processes
    multiprocessing.freeze_support()
    main()

