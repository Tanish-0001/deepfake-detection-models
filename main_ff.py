import os
import cv2
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import multiprocessing

from src.model_core import Two_Stream_Net


class FaceForensicsDataset(Dataset):
	"""
	PyTorch Dataset for FaceForensics++ style folder structure available under ./FF

	Expected structure (subset shown):
	FF/
	  ├── manipulated_sequences/
	  │   ├── DeepFakeDetection/
	  │   │   └── c23/videos/*
	  │   ├── Deepfakes/
	  │   │   └── c23/videos/*
	  │   ├── Face2Face/
	  │   │   └── c23/videos/*
	  │   └── ...
	  └── original_sequences/
		  ├── actors/c23/videos/*
		  └── youtube/c23/videos/*

	Notes:
	- The "videos" directory may contain video files (e.g., .mp4, .avi) OR subfolders with extracted frames.
	- Labels follow main_celeb_df.py convention: real=1, fake=0.
	"""

	VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
	IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

	def __init__(
		self,
		root_dir: str = "./FF",
		split: str = "train",
		transform=None,
		train_ratio: float = 0.8,
		seed: int = 42,
		frame_selection: str = "random",  # "random" | "middle"
	) -> None:
		super().__init__()
		self.root_dir = root_dir
		self.split = split
		self.transform = transform
		self.train_ratio = train_ratio
		self.seed = seed
		self.frame_selection = frame_selection

		# Discover samples
		all_samples: List[Tuple[str, int, str]] = []  # (path, label, type: "video"|"frames")

		# Real videos
		for real_sub in [
			os.path.join("original_sequences", "actors", "c23", "videos"),
			os.path.join("original_sequences", "youtube", "c23", "videos"),
		]:
			real_dir = os.path.join(root_dir, real_sub)
			if os.path.isdir(real_dir):
				all_samples.extend(self._gather_entries(real_dir, label=1))

		# Fake videos (manipulated)
		manipulated_root = os.path.join(root_dir, "manipulated_sequences")
		if os.path.isdir(manipulated_root):
			for method in os.listdir(manipulated_root):
				method_videos = os.path.join(manipulated_root, method, "c23", "videos")
				if os.path.isdir(method_videos):
					all_samples.extend(self._gather_entries(method_videos, label=0))

		if not all_samples:
			raise RuntimeError(
				f"No data found under {root_dir}. Expected FaceForensics++ structure as described in the docstring."
			)

		# Deterministic shuffle and split
		rnd = random.Random(seed)
		rnd.shuffle(all_samples)
		split_idx = int(len(all_samples) * train_ratio)
		if split == "train":
			self.samples = all_samples[:split_idx]
		else:
			self.samples = all_samples[split_idx:]

		print(
			f"[FaceForensicsDataset] Loaded {len(self.samples)} {split} samples "
			f"(total={len(all_samples)}, train_ratio={train_ratio})."
		)

		# Default transform
		if self.transform is None:
			self.transform = transforms.Compose(
				[
					transforms.ToPILImage(),
					transforms.Resize((256, 256)),
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
				]
			)

	def _gather_entries(self, path: str, label: int) -> List[Tuple[str, int, str]]:
		entries: List[Tuple[str, int, str]] = []
		try:
			for name in os.listdir(path):
				full = os.path.join(path, name)
				if os.path.isfile(full) and name.lower().endswith(self.VIDEO_EXTS):
					entries.append((full, label, "video"))
				elif os.path.isdir(full):
					# Could be a directory of frames; ensure it contains images
					if any(
						f.lower().endswith(self.IMAGE_EXTS)
						for f in os.listdir(full)
					):
						entries.append((full, label, "frames"))
		except FileNotFoundError:
			pass
		return entries

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int):
		path, label, ptype = self.samples[idx]

		if ptype == "video":
			if self.frame_selection == "middle":
				frame = self._read_middle_frame_from_video(path)
			else:
				frame = self._read_random_frame_from_video(path)
		else:
			if self.frame_selection == "middle":
				frame = self._read_middle_frame_from_folder(path)
			else:
				frame = self._read_random_frame_from_folder(path)

		if frame is None:
			raise RuntimeError(f"Failed to read a frame from {path}")

		# BGR -> RGB
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		frame = self.transform(frame)
		return frame, torch.tensor(label, dtype=torch.long)

	def _read_random_frame_from_video(self, video_path: str):
		cap = cv2.VideoCapture(video_path)
		if not cap.isOpened():
			return None
		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		if frame_count <= 0:
			# Fallback: just attempt to read first
			ret, frame = cap.read()
			cap.release()
			return frame if ret else None
		rand_idx = random.randint(0, max(0, frame_count - 1))
		cap.set(cv2.CAP_PROP_POS_FRAMES, rand_idx)
		ret, frame = cap.read()
		cap.release()
		return frame if ret else None

	def _read_random_frame_from_folder(self, frames_dir: str):
		imgs = [
			os.path.join(frames_dir, f)
			for f in os.listdir(frames_dir)
			if f.lower().endswith(self.IMAGE_EXTS)
		]
		if not imgs:
			return None
		img_path = random.choice(imgs)
		frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
		return frame

	def _read_middle_frame_from_video(self, video_path: str):
		cap = cv2.VideoCapture(video_path)
		if not cap.isOpened():
			return None
		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		if frame_count <= 0:
			ret, frame = cap.read()
			cap.release()
			return frame if ret else None
		mid_idx = max(0, frame_count // 2)
		cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
		ret, frame = cap.read()
		cap.release()
		return frame if ret else None

	def _read_middle_frame_from_folder(self, frames_dir: str):
		imgs = [
			os.path.join(frames_dir, f)
			for f in os.listdir(frames_dir)
			if f.lower().endswith(self.IMAGE_EXTS)
		]
		if not imgs:
			return None
		imgs.sort()
		img_path = imgs[len(imgs) // 2]
		frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
		return frame


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

	labels_np = np.array(labels_all)
	preds_np = np.array(preds_all)
	val_real_ratio = (labels_np == 1).mean()
	pred_real_ratio = (preds_np == 1).mean()
	print(f"[VAL CHECK] label real ratio: {val_real_ratio:.4f}, pred real ratio: {pred_real_ratio:.4f}, unique preds: {np.unique(preds_np, return_counts=True)}")

	epoch_loss = running_loss / len(dataloader.dataset)
	epoch_acc = accuracy_score(labels_all, preds_all)
	bal_acc = balanced_accuracy_score(labels_all, preds_all)
	f1_macro = f1_score(labels_all, preds_all, average="macro", zero_division=0)
	f1_pos = f1_score(labels_all, preds_all, pos_label=1, average="binary", zero_division=0)
	print(f"[VAL METRICS] acc: {epoch_acc:.4f} | bal_acc: {bal_acc:.4f} | f1_macro: {f1_macro:.4f} | f1_pos(=real): {f1_pos:.4f}")
	return epoch_loss, epoch_acc


def main():
	# Dataset / dataloaders
	train_ds = FaceForensicsDataset(root_dir="./FF", split="train", frame_selection="random")
	test_ds = FaceForensicsDataset(root_dir="./FF", split="test", frame_selection="middle")

	# Compute class distribution on train set for weighting and sampling
	train_labels = [lbl for _, lbl, _ in train_ds.samples]
	n_total = len(train_labels)
	n_pos = sum(1 for l in train_labels if l == 1)
	n_neg = n_total - n_pos
	print(f"[TRAIN SPLIT] total={n_total}, real(1)={n_pos} ({(n_pos/max(1,n_total))*100:.2f}%), fake(0)={n_neg} ({(n_neg/max(1,n_total))*100:.2f}%)")

	# WeightedRandomSampler to balance batches
	class_count = {0: max(1, n_neg), 1: max(1, n_pos)}
	class_weights = {c: n_total / class_count[c] for c in class_count}
	sample_weights = [class_weights[lbl] for lbl in train_labels]
	sampler = WeightedRandomSampler(weights=sample_weights, num_samples=n_total, replacement=True)

	# Use a modest batch size; adjust to your GPU memory
	train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler, shuffle=False, num_workers=4, pin_memory=True)
	test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

	# Device and model
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = Two_Stream_Net().to(device)

	# Optionally resume training if a checkpoint exists
	ckpt_dir = "./checkpoints_ff"
	os.makedirs(ckpt_dir, exist_ok=True)
	ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
	if os.path.exists(ckpt_path):
		try:
			model.load_state_dict(torch.load(ckpt_path, weights_only=True))
			print(f"Loaded existing checkpoint from {ckpt_path}")
		except Exception:
			print("Found checkpoint but could not load state dict. Starting fresh.")

	# Class-weighted loss to address imbalance
	weights_tensor = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float)
	weights_tensor = weights_tensor.to("cuda" if torch.cuda.is_available() else "cpu")
	criterion = nn.CrossEntropyLoss(weight=weights_tensor)  # Binary classification (real=1, fake=0)
	optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

	num_epochs = 10
	best_val_acc = 0.0

	for epoch in range(num_epochs):
		print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

		train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
		val_loss, val_acc = validate(model, test_loader, criterion, device)

		print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
		print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			torch.save(model.state_dict(), ckpt_path)
			print(f"✅ Saved new best model (Val Acc = {val_acc:.4f}) to {ckpt_path}")

		scheduler.step()

	print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
	multiprocessing.freeze_support()
	main()

