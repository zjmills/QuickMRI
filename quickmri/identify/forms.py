from django import forms


class MRIUploadForm(forms.Form):
    mri_image = forms.ImageField(label='Upload MRI Image')
