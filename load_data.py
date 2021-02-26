
from keras.datasets import cifar10

# load train and test dataset
def load_data():
	# load dataset
	(x_train, train_lable), (x_test, test_lable) = cifar10.load_data()
	
    # Normalise pixels
	x_train = x_train.astype('float32') / 255.0
	x_test = x_test.astype('float32') / 255.0

	return x_train, x_test, train_lable, test_lable