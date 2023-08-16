from network import Convolutional
import torch
from PIL import Image, ImageOps
import numpy as np
from matplotlib import pyplot as plt

model = Convolutional()
model.load_state_dict(torch.load('model.pt'))
model.eval()

image = ImageOps.grayscale(Image.open('example_healthy.jpg'))
image = image.resize((256, 256))
image = np.array(image)
image_torch = np.expand_dims(image, axis=0)
image_torch = np.expand_dims(image_torch, axis=0)

image_torch = torch.from_numpy(image_torch).float()
with torch.no_grad():
	output = model(image_torch)
	print(output)
	prediction = torch.round(output)
	print(prediction)

plt.imshow(image, cmap="gray")
plt.axis("off")
if int(prediction) == 0:
	plt.title('Prediction: Healthy')
else:
	plt.title('Prediction: Torn')
plt.show()
