import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt('valid-acl.csv', delimiter=',', dtype=np.int)

labels = []
idx = []

for i in data:
	labels.append(i[1])
	idx.append(i[2])

labels = np.array(labels)
idx = np.array(idx)

for i in range(120):
	file = np.load(f'valid/sagittal/{i+1130:04d}.npy')

	plt.imshow(file[idx[i]], cmap='gray')

	if labels[i] == 0:
		plt.title(f'Healthy: {i+1130:04d}: {idx[i]}')
	else:
		plt.title(f'Torn: {i+1130:04d}: {idx[i]}')

	plt.axis('off')
	plt.show()
