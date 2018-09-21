import tensorflow as tf
import numpy as np

from load_mnist_data import MNISTDataSampler
from image_utils import save_image

class GANModel():
	def __init__(self):
		self.generator_input_dim = 100
		self.generator_hidden_dim = 256
		self.generator_output_dim = 784

		self.discriminator_hidden_dim = 256
		self.discriminator_output_dim = 1

		self.generator_parameters = self.add_generator_parameters()
		self.discriminator_parameters = self.add_discriminator_parameters()

		self.add_loss_ops()
		self.add_training_ops()


	def add_generator_parameters(self):
		# input noise placeholder
		self.Z = tf.placeholder(tf.float32, shape=[None, self.generator_input_dim], name='Z')
		
		# hidden layer weights and biases
		self.G_W1 = tf.get_variable(
			name='G_W1',
			shape=[self.generator_input_dim, self.generator_hidden_dim],
			initializer=tf.contrib.layers.xavier_initializer()
		)

		self.G_b1 = tf.get_variable(
			name='G_b1',
			shape=[self.generator_hidden_dim],
			initializer=tf.zeros_initializer()
		)

		# output generation layer
		self.G_W2 = tf.get_variable(
			name='G_W2',
			shape=[self.generator_hidden_dim, self.generator_output_dim],
			initializer=tf.contrib.layers.xavier_initializer()
		)

		self.G_b2 = tf.get_variable(
			name='G_b2',
			shape=[self.generator_output_dim],
			initializer=tf.zeros_initializer()
		)

		return [self.G_W1, self.G_b1, self.G_W2, self.G_b2]

	
	def generator(self, Z):
		# first hidden layer
		G_h1 = tf.nn.relu(tf.matmul(Z, self.G_W1) + self.G_b1)
		
		# pre-activated output layer
		G_a2 = tf.matmul(G_h1, self.G_W2) + self.G_b2

		# output
		self.G_output = tf.nn.tanh(G_a2)
		return self.G_output


	def add_discriminator_parameters(self):
		# input image palceholder
		self.X = tf.placeholder(tf.float32, shape=[None, self.generator_output_dim], name='X')

		# hidden layer weights and biases
		self.D_W1 = tf.get_variable(
			name='D_W1',
			shape=[self.generator_output_dim, self.discriminator_hidden_dim],
			initializer=tf.contrib.layers.xavier_initializer()
		)

		self.D_b1 = tf.get_variable(
			name='D_b1',
			shape=[self.discriminator_hidden_dim],
			initializer=tf.zeros_initializer()
		)

		# output layer weights and biases
		self.D_W2 = tf.get_variable(
			name='D_W2',
			shape=[self.discriminator_hidden_dim, self.discriminator_output_dim],
			initializer=tf.contrib.layers.xavier_initializer()
		)

		self.D_b2 = tf.get_variable(
			name='D_b2',
			shape=[self.discriminator_output_dim],
			initializer=tf.zeros_initializer()
		)

		return [self.D_W1, self.D_b1, self.D_W2, self.D_b2]


	def discriminator(self, X):
		# hidden layer
		D_h1 = tf.nn.relu(tf.matmul(X, self.D_W1) + self.D_b1)

		D_logit = tf.matmul(D_h1, self.D_W2) + self.D_b2
		D_prob = tf.nn.sigmoid(D_logit)
		return D_prob


	def add_loss_ops(self):
		generated_samples = self.generator(self.Z)

		D_real = self.discriminator(self.X)
		D_fake = self.discriminator(generated_samples)

		discriminator_losses = tf.concat([tf.log(D_real), tf.log(1 - D_fake)], axis=-1)
		self.D_loss = - tf.reduce_mean(discriminator_losses)
		self.G_loss = - tf.reduce_mean(tf.log(D_fake))


	def add_training_ops(self):
		self.D_optimizer = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.discriminator_parameters)
		self.G_optimizer = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=self.generator_parameters)


	def generate_random_noise(self, minibatch_size):
		return np.random.normal(loc=0, scale=1, size=[minibatch_size, self.generator_input_dim])


	def train(self, iterations=100000, minibatch_size=32):
		# initialize session
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())

		data_sampler = MNISTDataSampler()

		for iteration in range(iterations):
			# train discriminator
			real_samples = data_sampler.get_random_images(minibatch_size)
			random_noise = self.generate_random_noise(minibatch_size)

			_, D_loss = sess.run(
							[self.D_optimizer, self.D_loss], 
							feed_dict={self.X: real_samples, self.Z: random_noise}
						)

			# train generator
			for _ in range(1):
				random_noise = self.generate_random_noise(minibatch_size)
				_, G_loss, G_output, G1_weights, G2_weights = sess.run(
										[self.G_optimizer, self.G_loss, self.G_output, self.G_W1, self.G_W2],
										feed_dict={self.Z: random_noise}
							   		  )

			if iteration % 100 == 0:
				print("Iteration {0}: discriminator loss = {1}, generator loss = {2}".format(
					iteration + 1,
					str(round(D_loss, 3)),
					str(round(G_loss, 3))
				))

			if iteration % 1000 == 0:
				save_image(G_output[0], "visualization", "output_iter-{0}.png".format(iteration + 1))








