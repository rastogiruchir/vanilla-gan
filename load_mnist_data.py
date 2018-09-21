import random
import tensorflow as tf 
mnist = tf.keras.datasets.mnist

from image_utils import preprocess_image

class MNISTDataSampler():
	def __init__(self):
		self.samples = self.load_data()
		self.shape = self.samples[0].shape
		

	def load_data(self):
		samples = []
		(x_train, y_train), (x_test, y_test) = mnist.load_data()

		samples = list(x_train)
		samples.extend(x_test)
		return samples


	def get_shape(self):
		return self.shape


	def get_random_image(self):
		num_samples = len(self.samples)
		random_index = random.randint(0, num_samples - 1)
		
		unprocessed_image = self.samples[random_index].flatten()
		return preprocess_image(unprocessed_image)


	def get_random_images(self, N):
		return [self.get_random_image() for _ in range(N)]
		

if __name__ == '__main__':
	mnist_data = MNISTDataSampler()
	
	unflattened_image = mnist_data.get_random_image(flattened=False)
	print("Shape of unflattened image: {0}".format(unflattened_image.shape))

	flattened_image = mnist_data.get_random_image(flattened=True)
	print("Shape of flattened image: {0}".format(flattened_image.shape))


