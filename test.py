import numpy as np
from matplotlib import pyplot as plt

file = np.load('train/sagittal/0000.npy')

# 21

plt.imshow(file[35], cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
