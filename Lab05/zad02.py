
from keras.applications.vgg16 import VGG16
from keras.models import Model
from csv import list_dialects
from os import listdir
from os import makedirs
from numpy import asarray
from numpy import save
from random import seed
from random import random
from shutil import copyfile
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Dense
from keras.layers import Flatten
import sys
from matplotlib import pyplot
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import gradient_descent_v2
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


folder ='df/'
photos, labels = list(), list()
for file in listdir(folder):
    output =0.0
    if file.startswith('dog'):
        output=1.0
    photo = load_img(folder + file, target_size=(200,200))
    photo = img_to_array(photo)
    photos.append(photo)
    labels.append(output)
photos= asarray(photos)
labels = asarray(labels)
print(photos.shape, labels.shape)
save('dogs_vs_cats_photos.npy', photos)
save('dogs_vs_cats_labels.npy', labels)
dataset_home = 'df/'
subdirs=['train/', 'test/']
for subdir in subdirs:
    labeldirs = ['dogs/', 'cats/']
    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir
        makedirs(newdir, exist_ok=True)
seed(1)
val_ratio = 0.2
src_directory = 'df/'
for file in listdir(src_directory):
    src = src_directory + '/' + file
    dst_dir = 'train/'
    if random() < val_ratio:
        dst_dir = 'test/'
    if file.startswith('cat'):
        dst = dataset_home + dst_dir + 'cats/' + file
        copyfile(src, dst)
    elif file.startswith('dog'):
        dst = dataset_home + dst_dir + 'dogs/' + file
        copyfile(src, dst)

def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(1, activation='sigmoid'))
	opt = gradient_descent_v2.SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

def summarize_diagnostics(history):
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.show()
	pyplot.close()

def run_test_harness():
	model = define_model()
	datagen = ImageDataGenerator(rescale=1.0/255.0)
	train_it = datagen.flow_from_directory('df/train/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	test_it = datagen.flow_from_directory('df/test/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
		validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
	print('> %.3f' % (acc * 100.0))
	summarize_diagnostics(history)

run_test_harness()

def define_model():
	model = VGG16(include_top=False, input_shape=(224, 224, 3))
	for layer in model.layers:
		layer.trainable = False
	flat1 = Flatten()(model.layers[-1].output)
	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
	output = Dense(1, activation='sigmoid')(class1)
	model = Model(inputs=model.inputs, outputs=output)
	opt = gradient_descent_v2.SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

def run_test_harness():
	model = define_model()
	datagen = ImageDataGenerator(featurewise_center=True)
	datagen.mean = [123.68, 116.779, 103.939]
	train_it = datagen.flow_from_directory('df/',
		class_mode='binary', batch_size=64, target_size=(224, 224))
	model.fit_generator(train_it, steps_per_epoch=len(train_it), epochs=10, verbose=0)
	model.save('final_model.h5')

run_test_harness()

def load_image(filename):
	img = load_img(filename, target_size=(224, 224))
	img = img_to_array(img)
	img = img.reshape(1, 224, 224, 3)
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img

def run_example():
	img = load_image('dog.jpg')
	model = load_model('final_model.h5')
	result = model.predict(img)
	print(result[0])

run_example()
