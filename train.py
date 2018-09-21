from gan import GANModel

def main():
	model = GANModel()
	model.train(minibatch_size=16)

if __name__ == '__main__':
	main()