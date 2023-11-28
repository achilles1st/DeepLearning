import keras.datasets as tfd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split






class CNN:
    def __init__(self, train_data, test_data, dict):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = train_data, test_data
        self.x_train, self.x_test = np.mean(self.x_train, axis=3), np.mean(self.x_test, axis=3)
        self.x_perturb, self.y_perturb = dict["x_perturb"], dict["y_perturb"]
        self.x_perturb = np.mean(self.x_perturb, axis=3)


    def execute_a(self):
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x_train, self.y_train, test_size=0.25, random_state=42, stratify = self.y_train
        )

        print(self.x_train.shape)
        print(self.y_train.shape)

        print(self.x_test.shape)
        print(self.y_test.shape)

        unique = []

        for i in range(20):
            unique.append(np.where(self.y_train == i)[0][0])

        plt.figure(figsize=(12, 8))
        for index in range(20):
            plt.subplot(5, 4, index + 1)
            plt.subplots_adjust(hspace=0.5)
            plt.imshow(self.x_train[unique[index]], cmap='gray')
            plt.title(self.y_train[unique[index]])
        plt.show()




if __name__ == '__main__':
    dict = pickle.load(open("C:/Users/nikla/Uni/Mit_Stef/DeepLearning/cifar20_perturb_test.pkl", "rb"))
    train_data, test_data = tfd.cifar100.load_data(label_mode="coarse")
    model_instance = CNN(train_data, test_data, dict)
    model_instance.execute_a()
