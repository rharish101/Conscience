#!/bin/python2
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, TensorBoard
from keras import backend as K
from scipy.misc import imread, imresize
from cv2 import cvtColor, COLOR_GRAY2RGB
import os
import sys
import time
from shutil import copy

def f1_score(y_true, y_pred):
    """Function to create Keras metric for F1 score
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    p = true_positives / (predicted_positives + K.epsilon())
    r = true_positives / (possible_positives + K.epsilon())
    f_score = 2 * (p * r) / (p + r + K.epsilon())
    return f_score

def load_images(shape=(224, 224, 3)):
    """Function to load images from disk

    This function loads images from the directory '256_ObjectClassification',
    which should be present in the same directory as this file. The directory
    structure of this directory should be the same as that of the zip file
    provided by The Programming Club, IIT Kanpur.

    The function saves the numpy array of images and labels after shuffling
    the dataset into two files: 'q1_images.npy' for the images and
    'q1_labels.npy' for the corresponding labels.

    Args:
        shape (tuple): The shape to which each image should be resized. Shape
                       must be a 3-dimensional tuple.

    """
    try:
        assert len(shape) == 3
    except AssertionError:
        print "Please provide 3 dimensional shape for input image with last "\
              "dimension as number of channels (3 for RGB)"

    num_images = 0
    if '256_ObjectClassification' not in os.listdir('.'):
        print "Please extract dataset into a folder named "\
              "'256_ObjectClassification'"
        return
    for folder in os.listdir('256_ObjectClassification'):
        for image in os.listdir('256_ObjectClassification/' + folder):
            if image[-4:] == '.jpg':
                num_images += 1

    data = np.zeros((num_images, shape[0], shape[1], shape[2]), np.float16)
    labels = np.zeros((num_images, 257), np.float16)
    counter = 0
    print "Loading data..."
    for folder in os.listdir('256_ObjectClassification'):
        for image in os.listdir('256_ObjectClassification/' + folder):
            if image[-4:] == '.jpg':
                img = imresize(imread('256_ObjectClassification/' + folder + '/' +\
                                    image), shape)
                if len(img.shape) == 2:
                    img = cvtColor(img, COLOR_GRAY2RGB)
                data[counter] = img
                labels[counter] = np.identity(257)[int(folder) - 1]
                counter += 1
                sys.stdout.write("\r%d images loaded" % counter)
                sys.stdout.flush()
    print ""

    combined = zip(data, labels)
    np.random.shuffle(combined)
    data, labels = zip(*combined)
    data = np.array(data)
    labels = np.array(labels)

    print "Saving data to disk..."
    np.save(open('q1_images.npy', 'w'), data)
    np.save(open('q1_labels.npy', 'w'), labels)
    print "Data saved"

def save_bottleneck_features(data=None, labels=None, input_shape=None,
                             batch_size=16, validation_split=0.3,
                             data_augment=False, data_aug_len=None):
    """Function to save features output by VGG16 convolutional layers

    This function takes input data and labels, splits into training and cross-
    validation data, augments them if necessary, and then saves the predictions
    of VGG16's convolutional layers (VGG16 with no dense layers) and labels.

    The function obtains input image data and labels from the disk if it has
    already been saved by the function: load_images()

    The function saves the training data features as 'q1_bottleneck_train.npy'
    and the labels as 'q1_labels_train.npy', and saves validation data as
    'q1_bottleneck_val.npy' and the labels as 'q1_labels_val.npy'
    
    Args:
        data (numpy.ndarray): The numpy array of input images
        labels (numpy.ndarray): The numpy array of input labels
        input_shape (tuple): The tuple for the input shape of each input image
                             Only used if data and labels are not provided
        batch_size (int): The batch size for generating predictions
        validation_split (float): A number between 0 and 1; the fraction of
                                  training data to be used for cross-validation
        data_augment (bool): True, if data is to be augmented
        data_aug_len (int): Length of total training set after augmentation

    """
    if (data is None) or (labels is None):
        if 'q1_images.npy' not in os.listdir('.') or 'q1_labels.npy' not in\
        os.listdir('.'):
            load_images(shape=shape)
        print "Loading image data from disk..."
        data = np.load('q1_images.npy', mmap_mode='r')
        labels = np.load('q1_labels.npy', mmap_mode='r')
    if data_augment and data_aug_len is None:
        data_aug_len = len(data)

    if data_augment:
        datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                                    height_shift_range=0.1,# rescale=1./255,
                                    shear_range=0.1, zoom_range=0.1,
                                    horizontal_flip=True, fill_mode='nearest')

    model = VGG16(include_top=False, weights='imagenet', input_shape=(
                data.shape[1], data.shape[2], data.shape[3]))
    train_data_len = int(len(data) * (1 - validation_split))
    if data_augment:
        data_aug_train_len = int(data_aug_len * (1 - validation_split))

    # For training data
    print "Calculating training data bottleneck features..."
    if data_augment:
        generator = datagen.flow(data[:train_data_len],
                                 y=labels[:train_data_len],
                                 batch_size=batch_size, shuffle=False)
        bottleneck_train = np.zeros(((data_aug_train_len // batch_size) * \
                                     batch_size, model.output_shape[1],
                                     model.output_shape[2],
                                     model.output_shape[3]), np.float16)
        labels_train = np.zeros(((data_aug_train_len // batch_size) * \
                                 batch_size, 257), np.float16)
        counter = 0
        for batch_x, batch_y in generator:
            try:
                bottleneck_train[counter * batch_size : (counter + 1) * \
                                batch_size] = model.predict_on_batch(batch_x)
                labels_train[counter * batch_size : (counter + 1) * batch_size]\
                = batch_y
                counter += 1
                sys.stdout.write("\rBatch %4d done" % (counter + 1))
                sys.stdout.flush()
                if counter >= (data_aug_train_len // batch_size):
                    break
            except ValueError:
                continue
    else:
        bottleneck_train = model.predict(data[:train_data_len],
                                         batch_size=batch_size, verbose=1)
    print "\nSaving training data bottleneck features..."
    np.save(open('q1_bottleneck_train.npy', 'w'), bottleneck_train)
    del bottleneck_train
    if not data_augment:
        labels_train = labels[:train_data_len]
    np.save(open('q1_labels_train.npy', 'w'), labels_train)
    del labels_train

    # For validation data
    print "Calculating validation data bottleneck features..."
    if data_augment:
        generator = datagen.flow(data[train_data_len:],
                                 y=labels[train_data_len:],
                                 batch_size=batch_size, shuffle=False)
        bottleneck_val = np.zeros((((data_aug_len - data_aug_train_len) // \
                                    batch_size) * batch_size,
                                    model.output_shape[1],
                                    model.output_shape[2],
                                    model.output_shape[3]), np.float16)
        labels_val = np.zeros((((data_aug_len - data_aug_train_len) // \
                                  batch_size) * batch_size, 257), np.float16)
        counter = 0
        for batch_x, batch_y in generator:
            try:
                bottleneck_val[counter * batch_size : (counter + 1) * batch_size]\
                = model.predict_on_batch(batch_x)
                labels_val[counter * batch_size : (counter + 1) * batch_size]\
                = batch_y
                counter += 1
                sys.stdout.write("\rBatch %4d done" % (counter + 1))
                sys.stdout.flush()
                if counter >= ((data_aug_len - data_aug_train_len) // batch_size):
                    break
            except ValueError:
                continue
    else:
        bottleneck_val = model.predict(data[train_data_len:],
                                       batch_size=batch_size, verbose=1)
    print "\nSaving validation data bottleneck features..."
    np.save(open('q1_bottleneck_val.npy', 'w'), bottleneck_val)
    del bottleneck_val
    if not data_augment:
        labels_val = labels[train_data_len:]
    np.save(open('q1_labels_val.npy', 'w'), labels_val)
    del labels_val

    print "Bottleneck features saved"

def train_dense(bottleneck_train=None, labels_train=None, bottleneck_val=None,
                labels_val=None, data=None, labels=None, batch_size=16,
                epochs=500, validation_split=0.3, input_shape=(224, 224, 3),
                data_augment=False, data_aug_len=None):
    """Function for generating a fully-connected neural network model

    This function trains a fully-connected model that is to be added to VGG16
    in place of VGG16's fully-connected layers. The model is saved as 
    'q1_dense_top.h5'
    
    The function automatically loads the input VGG16 training and validation
    features and labels from disk if it has already been saved by the function:
    save_bottleneck_features()

    Args:
        bottleneck_train (numpy.ndarray): The features generated by VGG16 on
                                          the training set
        labels_train (numpy.ndarray): The labels for the training set
        bottleneck_val (numpy.ndarray): The features generated by VGG16 on the
                                        cross-validation set
        labels_val (numpy.ndarray): The labels for the cross-validation set
        batch_size (int): The batch size for training the neural network
        epochs (int): The number of epochs for which to train the network
        data (numpy.ndarray): The input image numpy array, if VGG16 features
                              are not provided
        labels (numpy.ndarray): The input image labels numpy array, if VGG16
                                features are not provided
        validation_split (float): A number between 0 and 1; the fraction of
                                  training data to be used for validation,
                                  if VGG16 features are not provided
        data_augment (bool): True, if data is to be augmented, if VGG16
                             features are not provided
        data_aug_len (int): Length of total training set after augmentation,
                            if VGG16 features are not provided
        input_shape (tuple): The shape to which each image should be resized,
                             if no input data is provided. Shape must be a
                             3-dimensional tuple.

    Returns:
        keras.models.Sequential: A Sequential Keras model instance

    """
    if bottleneck_train is None or labels_train is None or bottleneck_val is\
    None or labels_val is None:
        if 'q1_bottleneck_train.npy' not in os.listdir('.') or\
        'q1_labels_train.npy' not in os.listdir('.') or\
        'q1_bottleneck_train.npy' not in os.listdir('.') or\
        'q1_labels_val.npy' not in os.listdir('.'):
            save_bottleneck_features(data=data, labels=labels,
                                     batch_size=batch_size,
                                     validation_split=validation_split,
                                     input_shape=input_shape,
                                     data_augment=data_augment,
                                     data_aug_len=data_aug_len)
        print "Loading data from disk..."
        bottleneck_train = np.load('q1_bottleneck_train.npy',
                                   mmap_mode='r')
        labels_train = np.load('q1_labels_train.npy', mmap_mode='r')
        bottleneck_val = np.load('q1_bottleneck_val.npy',
                                 mmap_mode='r')
        labels_val = np.load('q1_labels_val.npy', mmap_mode='r')

    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_train.shape[1:]))
    model.add(Dense(256, kernel_initializer='glorot_normal',
                    bias_initializer='glorot_normal', 
                    activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(257, kernel_initializer='glorot_normal',
                    bias_initializer='glorot_normal',
                    activation='softmax'))

    model.compile(optimizer='sgd', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='loss', min_delta=0.01, patience=5)
    now = str(time.time())
    tensorboard = TensorBoard(log_dir='./Tensorboard/q1/' + now)

    if bottleneck_val.shape == (0,):
        model.fit(bottleneck_train, labels_train, epochs=500, batch_size=16,
                  callbacks=[early_stop, tensorboard])
    else:
        model.fit(bottleneck_train, labels_train, epochs=500, batch_size=16,
                  validation_data=(bottleneck_val, labels_val), callbacks=[
                  early_stop, tensorboard])
    response = raw_input("Do you want to save this model? (Y/n): ")
    if response.lower() not in ['n', 'no', 'nah', 'nein', 'nahi', 'nope']:
        model.save('q1_dense_top.h5')
        copy('./q1.py', './Tensorboard/' + now)
        print "Dense top layer saved"
    return model

def train_vgg_mod(data=None, labels=None, validation_split=0.3,
                  batch_size=32, epochs=500, data_augment=False,
                  data_aug_len=None, input_shape=(224, 224, 3),
                  bottleneck_train=None, labels_train=None,
                  bottleneck_val=None, labels_val=None):
    """Function for generating a fine-tuned VGG16 model

    This function trains a fine-tuned VGG16 model that is to be used for
    final prediction. The model is saved as 'q1_vgg_mod.h5'
    
    The function automatically loads the input image data and labels from disk
    if it has already been saved by the function: load_images(). It also
    automatically loads the fully-connected model trained generated by the
    function: train_dense()

    Args:
        data (numpy.ndarray): The input image numpy array
        labels (numpy.ndarray): The input image labels numpy array
        batch_size (int): The batch size for training the neural network
        epochs (int): The number of epochs for which to train the network
        validation_split (float): A number between 0 and 1; the fraction of
                                  training data to be used for validation
        data_augment (bool): True, if data is to be augmented
        data_aug_len (int): Length of total training set after augmentation
        bottleneck_train (numpy.ndarray): The features generated by VGG16 on
                                          the training set
        labels_train (numpy.ndarray): The labels for the training set
        bottleneck_val (numpy.ndarray): The features generated by VGG16 on the
                                        cross-validation set
        labels_val (numpy.ndarray): The labels for the cross-validation set
        input_shape (tuple): The shape to which each image should be resized,
                             if no input data is provided. Shape must be a
                             3-dimensional tuple.

    Returns:
        keras.models.Sequential: A Sequential Keras model instance

    """
    if (data is None) or (labels is None):
        if 'q1_images.npy' not in os.listdir('.') or 'q1_labels.npy' not in\
        os.listdir('.'):
            load_images(input_shape)
        print "Loading image data from disk..."
        data = np.load('q1_images.npy', mmap_mode='r')
        labels = np.load('q1_labels.npy', mmap_mode='r')
    if data_augment:
        datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                                     height_shift_range=0.1,# rescale=1./255,
                                     shear_range=0.1, zoom_range=0.1,
                                     horizontal_flip=True, fill_mode='nearest')
        train_data_len = int(len(data) * (1 - validation_split))
        generator_train = datagen.flow(data[:train_data_len],
                                       y=labels[:train_data_len],
                                       batch_size=batch_size, shuffle=False)
        generator_val = datagen.flow(data[train_data_len:],
                                     y=labels[train_data_len:],
                                     batch_size=batch_size, shuffle=False)
        data_aug_train_len = int(data_aug_len * (1 - validation_split))

    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(
                data.shape[1], data.shape[2], data.shape[3]))
    for layer in vgg.layers[:-4]:
        layer.trainable = False

    if 'q1_dense_top.h5' in os.listdir('.'):
        top_model = load_model('q1_dense_top.h5')
    else:
        top_model = train_dense(data=data, labels=labels, batch_size=batch_size,
                                epochs=epochs,
                                validation_split=validation_split,
                                bottleneck_train=bottleneck_train,
                                labels_train=labels_train,
                                bottleneck_val=bottleneck_val,
                                labels_val=labels_val,
                                data_augment=data_augment,
                                data_aug_len=data_aug_len)

    model = Sequential()
    model.add(vgg)
    model.add(top_model)
    opt = SGD(lr=5e-5, decay=0.9)
    early_stop = EarlyStopping(monitor='loss', min_delta=0.01, patience=5)
    now = str(time.time())
    tensorboard = TensorBoard(log_dir='./Tensorboard/q1/' + now)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy', f1_score])

    if data_augment:
        model.fit_generator(generator_train, epochs=epochs,
                            steps_per_epoch=(data_aug_train_len // batch_size),
                            validation_data=generator_val,
                            validation_steps=((data_aug_len -\
                            data_aug_train_len) // batch_size),
                            callbacks=[early_stop, tensorboard])
    else:
        model.fit(data, labels, epochs=epochs, batch_size=batch_size,
                  validation_split=validation_split, callbacks=[early_stop,
                  tensorboard])
    response = raw_input("Do you want to save this model? (Y/n): ")
    if response.lower() not in ['n', 'no', 'nah', 'nein', 'nahi', 'nope']:
        model.save('q1_vgg_mod.h5')
        copy('./q1.py', './Tensorboard/' + now)
    print "Modified VGG model saved"
    return model

def model_predict(shape=(224, 224, 3), vgg_model=None, batch_size=16):
    """Function to evaluate model performance on data

    Args:
        shape (tuple): The shape to which each image should be resized. Shape
                       must be a 3-dimensional tuple.
        vgg_model (keras.model.Model): A Keras model instance of the fine-
                                       tuned VGG model
        batch_size (int): The batch size for generating predictions

    """ 
    if vgg_model is None:
        if 'q1_vgg_mod.h5' in os.listdir('.'):
            vgg_model = load_model('q1_vgg_mod.h5', custom_objects={
                                   'f1_score':f1_score})
        else:
            vgg_model = train_vgg_mod()

    num_images = 0
    predictions = []
    counter = 0
    print "Loading data..."
    for image in os.listdir('validation_question1'):
        if image[-4:] == '.jpg':
            img = imresize(imread('validation_question1/' + image), shape)
            if len(img.shape) == 2:
                img = cvtColor(img, COLOR_GRAY2RGB)
            pred = vgg_model.predict(np.array([img,]), batch_size=1)
            predictions.append([np.argmax(pred) + 1, int(image[:-4])])
            counter += 1
            sys.stdout.write("\r%d image predictions done" % counter)
            sys.stdout.flush()
    print ""
    pred_file = open('q1_predictions_mauryans.txt', 'w')
    for i in range(1, len(predictions) + 1):
        for pred in predictions:
            if pred[1] == i:
                val = pred[0]
        pred_file.write(("%03d" % val[0]) + '\n')
    pred_file.close()
    print "Predictions saved"

if __name__ == '__main__':
    #load_images()
    #save_bottleneck_features(validation_split=0)
    #save_bottleneck_features(data_augment=True, data_aug_len=40000)
    #train_dense(validation_split=0)
    train_vgg_mod()
    model_predict()

