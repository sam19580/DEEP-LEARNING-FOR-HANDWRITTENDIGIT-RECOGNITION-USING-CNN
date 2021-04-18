# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 11:41:09 2021

@author: samanvay bhatt
"""

"""
The Keras API supports this by specifying the “validation_data” argument to the
 model.fit() function when training the model, that will, in turn, return an 
 object that describes model performance for the chosen loss and metrics on
 each training epoch.
"""
# deeper cnn model for mnist

from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import RMSprop
# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY
 
# scale pixels
"""
    We know that the pixel values for each image in the dataset are unsigned integers
    in the range between black and white, or 0 and 255.

    We do not know the best way to scale the pixel values for modeling, but we know
    that some scaling will be required.
     A good starting point is to normalize the pixel values of grayscale images, e.g. rescale
     them to the range [0,1]. This involves first converting the data type from unsigned integers
     to floats, then dividing the pixel values by the maximum value.
"""
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm
 
# define cnn model
"""
    For the convolutional front-end, we can start with a single convolutional layer with a small
    filter size (3,3) and a modest number of filters (32) followed by a max pooling layer. The 
    filter maps can then be flattened to provide features to the classifier.
    Given that the problem is a multi-class classification task, we know that we will require an
    output layer with 10 nodes in order to predict the probability distribution of an image belonging
    to each of the 10 classes. This will also require the use of a softmax activation function. Between
    the feature extractor and the output layer, we can add a dense layer to interpret the features, in 
    this case with 100 nodes.
"""
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = RMSprop(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
	return model
 
"""
    The model will be evaluated using five-fold cross-validation.
    The value of k=5 was chosen to provide a baseline for both repeated 
    evaluation and to not be so large as to require a long running time.
    Each test set will be 20% of the training dataset, or about 12,000 examples, 
    close to the size of the actual test set for this problem.
    The evaluate_model() function below implements these behaviors, taking the 
    training dataset as arguments and returning a list of accuracy scores and 
    training histories that can be later summarized.
"""

# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
	scores, histories = list(), list()
	# prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	for train_ix, test_ix in kfold.split(dataX):
		# define model
		model = define_model()
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model
		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
		# evaluate model
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# stores scores
		scores.append(acc)
		histories.append(history)
	return scores, histories
 
# plot diagnostic learning curves
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		pyplot.subplot(2, 1, 1)
		pyplot.title('Cross Entropy Loss')
		pyplot.plot(histories[i].history['loss'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		pyplot.subplot(2, 1, 2)
		pyplot.title('Classification Accuracy')
		pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	pyplot.show()
 
# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	pyplot.boxplot(scores)
	pyplot.show()
 
# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# evaluate model
	scores, histories = evaluate_model(trainX, trainY)
	# learning curves
	summarize_diagnostics(histories)
	# summarize estimated performance
	summarize_performance(scores)
 
# entry point, run the test harness
run_test_harness()
