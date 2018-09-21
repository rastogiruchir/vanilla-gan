import os
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
	''' Normalizes an array containing values between 0 and 255 to
		be between -1 and 1'''
	scaling_factor = float(255) / 2
	new_image = (image / scaling_factor) - 1
	return new_image


def postprocess_image(image):
	''' Normalizes an array containing values between -1 and 1
		to be between 0 and 255 '''
	scaling_factor = float(255) / 2
	new_image = (image + 1) * scaling_factor
	return new_image


def save_image(image, relative_dir, filename):
	postprocessed_image = postprocess_image(image)
	reshaped_image = np.reshape(postprocessed_image, [28, 28]).astype(np.uint8)
	plt.imshow(reshaped_image, cmap='gray')

	if not os.path.exists(relative_dir):
		os.makedirs(relative_dir)
		
	plt.savefig(os.path.join(relative_dir, filename))
