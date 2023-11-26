import keras.datasets as tfd
import numpy as np
import pickle





class CNN:
    def __init__(self, train_data, test_data, dict):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = train_data, test_data
        self.x_train, self.x_test = np.mean(self.x_train, axis=3), np.mean(self.x_test, axis=3)
        self.x_perturb, self.y_perturb = dict["x_perturb"], dict["y_perturb"]
        self.x_perturb = np.mean(self.x_perturb, axis=3)






if __name__ == '__main__':
    dict = pickle.load(open("cifar20_perturb_test.pkl", "rb"))
    train_data, test_data = tfd.cifar100.load_data(label_mode="coarse")
    model_instance = CNN(train_data, test_data, dict)
