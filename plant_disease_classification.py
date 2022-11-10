# --------------------------Python version used - 3.10.7 -------------------------- #
import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Activation,Flatten,Dense,Dropout

# plotting function (to plot graphs)
def plot_results_graphs(train):
    acc = train.history['accuracy']
    val_acc = train.history['val_accuracy']
    loss = train.history['loss']
    val_loss = train.history['val_loss']
    epochs = range(1, len(acc) + 1)
    #Train and validation accuracy
    plt.plot(epochs, acc, 'b', label='Training accurarcy')
    plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
    plt.title('Training and Validation accurarcy')
    plt.legend()
    plt.figure()
    #Train and validation loss
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # taking input path to training data
    parser.add_argument('--train',required=True,help="Path to training dataset")
    # taking input path to validation data
    parser.add_argument("--valid",required=True,help="Path to validation dataset")
    args=parser.parse_args()
    # train data
    train_data = args.train
    # valid data
    valid_data = args.valid

    # Image augmentation with the help of ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale = 1./255,brightness_range = [0.7,1.8],shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
    valid_datagen = ImageDataGenerator(rescale=1./255)
    batch_size=3
    train_generator = train_datagen.flow_from_directory(train_data,target_size=(256,256),batch_size = batch_size)
    valid_generator = valid_datagen.flow_from_directory(valid_data,target_size=(256,256),batch_size = batch_size)

    # Input shape of images
    input_shape = (256,256,3)
    number_of_classes = len(glob.glob(train_data+"/*"))

    # Model 
    model = Sequential()
    # Feature Extraction
    model.add(Conv2D(32,(3,3),input_shape = input_shape,activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,(3,3),activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128,(3,3),activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Artificial Neural Networks
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dense(number_of_classes,activation='softmax'))
    # model compilation
    model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=3,restore_best_weights=True)
    checkpoints = keras.callbacks.ModelCheckpoint('model{epoch:08d}.h5',period = 2)
    # training
    train = model.fit(train_generator,epochs=25,validation_data = valid_generator,verbose=1,callbacks=[callback,checkpoints])
    plot_results_graphs(train)
    # saving the model
    model.save("model_1.h5")
    # saving model weights
    model.save_weights("model_1_weights.h5")








    
