from network import Convolutional
import datasets
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import pickle

num_epochs = 100
learning_rate = 0.00025
patience = 4

train_dataset = datasets.train_dataset
valid_dataset = datasets.valid_dataset
train_loader = datasets.train_loader
valid_loader = datasets.valid_loader

model = Convolutional()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
loss_tracker = []
valid_loss_tracker = []
acc_tracker = []
patience_tracker = []
avg_loss_tracker = []

best_valid_loss = np.inf
current_patience = 0

for epoch in range(num_epochs):
	epoch_loss = 0
	num_iters = 0
	progress_bar = tqdm(enumerate(train_loader), total=n_total_steps, leave=False)
	for i, (images, labels) in progress_bar:
		images = images.unsqueeze(1)
		images = images.to(torch.float32)
		labels = labels.view(-1, 1)
		labels = labels.to(torch.float32)

		# forward
		outputs = model(images)

		# loss
		loss = criterion(outputs, labels)

		# backward
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		loss_tracker.append(loss.item())
		epoch_loss += loss.item()
		num_iters += 1

		progress_bar.set_description(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
		progress_bar.update()

	with torch.no_grad():
		n_correct = 0
		n_samples = 0
		valid_loss = 0

		for images, labels in valid_loader:
			images = images.unsqueeze(1)
			images = images.to(torch.float32)
			labels = labels.view(-1, 1)
			labels = labels.to(torch.float32)
			outputs = model(images)
			predicted = torch.round(outputs)

			valid_loss += criterion(outputs, labels).item()

			n_samples += labels.size(0)
			n_correct += (predicted == labels).sum().item()

		avg_loss_tracker.append(epoch_loss / num_iters)
		acc = 100.0 * n_correct / n_samples
		avg_valid_loss = valid_loss / len(valid_loader)
		valid_loss_tracker.append(avg_valid_loss)
		acc_tracker.append(acc)

		# Check for early stopping condition
		if avg_valid_loss < best_valid_loss:
			best_valid_loss = avg_valid_loss
			current_patience = 0
			patience_tracker.append(current_patience)
			torch.save(model.state_dict(), 'model.pt')
			print(f'\nAccuracy of the Network: {acc:.4f}%, Number of Correct Identifications: {n_correct}')
			print(f'Validation Loss: {avg_valid_loss:.4f}, Best Validation Loss: {best_valid_loss:.4f}')
			print(f'Current Patience: {current_patience}')
		else:
			current_patience += 1
			patience_tracker.append(current_patience)
			print(f'\nAccuracy of the Network: {acc:.4f}%, Number of Correct Identifications: {n_correct}')
			print(f'Validation Loss: {avg_valid_loss:.4f}, Best Validation Loss: {best_valid_loss:.4f}')
			print(f'Current Patience: {current_patience}')
			if current_patience >= patience:
				print(f'Early stopping at epoch {epoch + 1}')
				break

print('Finished Training')

model = Convolutional()
model.load_state_dict(torch.load('model.pt'))
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
	for images, labels in valid_loader:
		images = images.unsqueeze(1).to(torch.float32)
		labels = labels.view(-1, 1).to(torch.float32)
		outputs = model(images)
		predicted = torch.round(outputs)
		predictions.extend(predicted.cpu().numpy())
		true_labels.extend(labels.cpu().numpy())

fpr, tpr, thresholds = roc_curve(true_labels, predictions)
auc = roc_auc_score(true_labels, predictions)

print(predictions)
print(true_labels)

with torch.no_grad():
	n_correct = 0
	n_samples = 0
	for images, labels in valid_loader:
		images = images.unsqueeze(1)
		images = images.to(torch.float32)
		labels = labels.view(-1, 1)
		labels = labels.to(torch.float32)
		outputs = model(images)
		predicted = torch.round(outputs)
		n_samples += labels.size(0)
		n_correct += (predicted == labels).sum().item()

	acc = 100.0 * n_correct / n_samples

	print(f'Accuracy of the Network: {acc:.4f}%, Number of Correct Identifications: {n_correct}, AUC-ROC: {auc:.4f}')

variables = [loss_tracker, avg_loss_tracker, valid_loss_tracker, acc_tracker, patience_tracker, fpr, tpr, auc]

with open('objs.pkl', 'wb') as f:
	pickle.dump(variables, f)
