from django.http import HttpResponse
import torch
from django.shortcuts import render
from .models import Convolutional
from PIL import Image, ImageOps
from .forms import MRIUploadForm
import numpy as np
from django.conf import settings
from django.core.files.storage import FileSystemStorage

# Create your views here.

model = Convolutional()
model.load_state_dict(torch.load('identify/model.pt'))
model.eval()


def diagnose_page(request):
	if request.method == 'POST':
		form = MRIUploadForm(request.POST, request.FILES)

		if form.is_valid():
			mri_image = form.cleaned_data['mri_image']

			# Process the uploaded MRI image using the pre-trained model
			diagnosis = process_mri_image(mri_image)
			print(diagnosis)

			# Render the diagnose page template with the diagnosis result
			return render(request, 'diagnose.html', {'diagnosis': diagnosis})
	else:
		form = MRIUploadForm()

	# Render the diagnose page template with the empty form
	return render(request, 'diagnose.html', {'form': form})


def process_mri_image(mri_image):
	image = Image.open(mri_image)
	image = image.convert('L')
	image = image.resize((256, 256))
	image.save('static/identify/image.jpg')
	image = np.array(image)
	image_torch = np.expand_dims(image, axis=0)
	image_torch = np.expand_dims(image_torch, axis=0)
	image_torch = torch.from_numpy(image_torch).float()

	with torch.no_grad():
		output = model(image_torch)
	# Process the model output to obtain the diagnosis
	diagnosis = torch.round(output).item()  # Convert tensor to Python scalar

	return diagnosis
