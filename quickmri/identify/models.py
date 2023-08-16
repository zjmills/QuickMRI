from django.db import models
import torch
import torch.nn as nn
import torch.nn.functional as F

# Create your models here.


class Convolutional(nn.Module):
	def __init__(self):
		super(Convolutional, self).__init__()
		self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
		self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
		self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
		self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
		self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
		self.conv6 = nn.Conv2d(256, 512, 3, padding=1)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(512*4*4, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 1)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = self.pool(F.relu(self.conv3(x)))
		x = self.pool(F.relu(self.conv4(x)))
		x = self.pool(F.relu(self.conv5(x)))
		x = self.pool(F.relu(self.conv6(x)))
		x = x.view(-1, 512*4*4)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		x = torch.sigmoid(x)
		return x
