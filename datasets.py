import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TrainDataset(Dataset):
	def __init__(self):
		train_data = np.loadtxt('train-acl.csv', delimiter=',', dtype=np.int)

		labels = []
		idx = []

		for i in train_data:
			labels.append(i[1])
			idx.append(i[2])

		labels = np.array(labels)
		idx = np.array(idx)

		train_labels = []
		train_images = []

		for i in range(1130):
			file = np.load(f'train/sagittal/{i:04d}.npy')

			if idx[i] != -1:
				train_labels.append(labels[i])
				train_images.append(file[idx[i]])

		train_labels = np.array(train_labels)
		train_images = np.array(train_images)

		self.labels = torch.from_numpy(train_labels)
		self.images = torch.from_numpy(train_images)
		self.n_samples = train_images.shape[0]

	def __getitem__(self, index):
		return self.images[index], self.labels[index]

	def __len__(self):
		return self.n_samples


class ValidDataset(Dataset):
	def __init__(self):
		train_data = np.loadtxt('valid-acl.csv', delimiter=',', dtype=np.int)

		labels = []
		idx = []

		for i in train_data:
			labels.append(i[1])
			idx.append(i[2])

		labels = np.array(labels)
		idx = np.array(idx)

		train_labels = []
		train_images = []

		for i in range(120):
			file = np.load(f'valid/sagittal/{i+1130:04d}.npy')

			if idx[i] != -1:
				train_labels.append(labels[i])
				train_images.append(file[idx[i]])

		train_labels = np.array(train_labels)
		train_images = np.array(train_images)

		self.labels = torch.from_numpy(train_labels)
		self.images = torch.from_numpy(train_images)
		self.n_samples = train_images.shape[0]

	def __getitem__(self, index):
		return self.images[index], self.labels[index]

	def __len__(self):
		return self.n_samples


train_dataset = TrainDataset()
valid_dataset = ValidDataset()

print(valid_dataset.__len__())

train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=50, shuffle=False)
