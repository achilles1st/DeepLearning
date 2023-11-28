import keras.datasets as tfd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.optimizers import schedules
from keras.models import Sequential
from keras import layers, models
import tensorflow as tf
import random



class CNN:
    def __init__(self, train_data, test_data, perturb):
        # load dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = train_data, test_data
        # converting all images to grayscale via averaging the three RGB channel pixel values
        self.x_train, self.x_test_ = np.mean(self.x_train, axis=3), np.mean(self.x_test, axis=3)
        # load perturbed dataset
        self.x_perturb, self.y_perturb = perturb["x_perturb"], perturb["y_perturb"]
        # converting all images to grayscale via averaging the three RGB channel pixel values
        self.x_perturb = np.mean(self.x_perturb, axis=3)

        # normalize to range 0-1
        self.x_train, self.x_test = self.preprocess_data(self.x_train, self.x_test)

        # one hot encode target values
        self.y_train = tf.keras.utils.to_categorical(self.y_train, num_classes=20)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, num_classes=20)

        # split train data into train and validation data
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x_train, self.y_train, test_size=0.25, random_state=42, stratify=self.y_train
        )

    def preprocess_data(self, train, test):
        # convert from integers to floats
        train_norm = train.astype('float32')
        test_norm = test.astype('float32')
        # normalize to range 0-1
        train_norm = train_norm / 255.0
        test_norm = test_norm / 255.0
        # return normalized images
        return train_norm, test_norm

    def plot_samples(self, x_data, y_data, class_labels, num_samples=3):
        plt.figure(figsize=(20, 10))
        # loop over each class label and create a grid of num_samples x num_samples
        for i, label in enumerate(class_labels):
            class_indices = np.where(np.argmax(y_data, axis=1) == label)[0]
            samples = random.sample(list(class_indices), min(num_samples, len(class_indices)))
            # loop over each sample and plot it
            for j, idx in enumerate(samples):
                if num_samples == 1:
                    plt.subplot(5, 4, i + 1)
                else:
                    plt.subplot(len(class_labels), num_samples, i * num_samples + j + 1)

                plt.imshow(np.squeeze(x_data[idx]), cmap='gray')
                plt.title(f'Class {label}')
                plt.axis('off')

        plt.tight_layout()
        plt.savefig('figures/calsses.png')
        plt.show()

    def execute_a(self):

        print(self.x_train.shape)
        print(self.y_train.shape)

        print(self.x_test.shape)
        print(self.y_test.shape)
        # plot 3 random samples from each class (10 classes)
        self.plot_samples(self.x_train, self.y_train, class_labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
                          num_samples=1)


if __name__ == '__main__':
    perturb = pickle.load(open("cifar20_perturb_test.pkl", "rb"))
    train_data, test_data = tfd.cifar100.load_data(label_mode="coarse")
    model_instance = CNN(train_data, test_data, perturb)

    model_instance.execute_a()

