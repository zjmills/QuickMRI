import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt('train-acl.csv', delimiter=',', dtype=np.int)

labels = []
idx = []

for i in data:
	labels.append(i[1])
	idx.append(i[2])

labels = np.array(labels)
idx = np.array(idx)

for i in range(1130):
	file = np.load(f'train/sagittal/{i:04d}.npy')

	plt.imshow(file[idx[i]], cmap='gray')

	if labels[i] == 0:
		plt.title(f'Healthy: {i:04d}: {idx[i]}')
	else:
		plt.title(f'Torn: {i:04d}: {idx[i]}')

	plt.axis('off')
	plt.show()
