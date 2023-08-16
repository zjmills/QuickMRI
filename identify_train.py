import numpy as np
from matplotlib import pyplot as plt

for i in range(1130):
	file = np.load(f'train/sagittal/{i:04d}.npy')

	num_img = file.size / (256*256)
	print(num_img)

	idx = 0
	fig, axes = plt.subplots(6, 9, figsize=(25, 25))

	for j in range(6):
		for k in range(9):
			ax = axes[j, k]
			if idx < num_img:
				ax.imshow(file[idx], cmap='gray')
				ax.set_title(f'{i:04d}: {idx}')
			ax.axis('off')
			idx += 1

	plt.tight_layout()
	plt.show()
