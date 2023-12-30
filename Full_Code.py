import numpy as np
import tensorflow as tf
from morphomnist import io, morpho, perturb
import pandas as pd
from Regressions_Tests import test_images_regression
from synthetic_tests import synthetic_tests

def main():
    #classification_model_name = "Models/Cifar100_classification_model"
    regression_model_name = "Models/MorphoMnist_Regression_Model"
    synthetic_model_name = ["Models/Synthetic_Model_Periodic", "Models/Weierstrass"]
    #Load the data
    images,label,attributes=load_data_m_mnist()

    morpho_train = images[0:int(.9*len(images))]
    morpho_test = images[int(.9*len(images)):len(images)]

    morpho_train_labels = label[0:int(.9*len(images))]
    morpho_test_labels = label[int(.9*len(images)):len(images)]

    morpho_train_attributes = attributes[0:int(.9*len(images)),1:]
    morpho_test_attributes = attributes[int(.9*len(images)):len(images),1:]

    regularization_const = 10**(-1.75)
    #Get the model
    #test_images_classification(cifar_100_train_images, cifar_100_train[b'fine_labels'], cifar_100_test_images, cifar_100_test[b'fine_labels'],regularization_const,.95,classification_model_name,0)
    test_images_regression(morpho_train, morpho_train_attributes, morpho_test, morpho_test_attributes, regularization_const, .9,regression_model_name,previous_model=0)
    synthetic_tests(synthetic_function1,regularization_const, .9, 10000, -10, 10, 17, synthetic_model_name[0])
    synthetic_tests(synthetic_function2,regularization_const, .9, 10000, -10, 10, 17, synthetic_model_name[1])

def synthetic_function1(x):
    return np.sin(x)*np.log(np.abs(x)+.01)

def synthetic_function2(x):
    first_term = np.ones(np.shape(x))

    for i in range(100):
        first_term += .5**i*np.cos(10**i*np.pi*x/100)
    return first_term
def load_data_m_mnist():
    images = io.load_idx("Data/Morphomnist/train-images-idx3-ubyte.gz")
    labels = io.load_idx("Data/Morphomnist/train-labels-idx1-ubyte.gz")
    attributes=pd.read_csv("Data/Morphomnist/train-morpho.csv")
    return images/255,labels,pd.DataFrame.to_numpy(attributes)
'''def load_cifar():
    with open("Data/Cifar100/train", 'rb') as fo:
        dict_train = pickle.load(fo, encoding='bytes')
    with open("Data/Cifar100/test", 'rb') as fo:
        dict_test = pickle.load(fo, encoding='bytes')
    return dict_train,dict_test'''
if __name__ == '__main__':
    tf.random.set_seed(25)
    np.random.seed(25)
    main()