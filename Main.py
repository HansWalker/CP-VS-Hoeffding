import numpy as np
import tensorflow as tf
from morphomnist import io, morpho, perturb
import pandas as pd
from Models import get_classifier_image
from Classification_Tests import test_images_classification
import pickle

def main():
    classification_model_name = "Models/Cifar100_classification_model.h5"
    #Load the data
    images,label,attributes=load_data_m_mnist()

    cifar_100_train, cifar_100_test=load_cifar()

    cifar_100_train_images = np.reshape(cifar_100_train[b'data'],(np.shape(cifar_100_train[b'data'])[0],32,32,3))/255

    cifar_100_test_images = np.reshape(cifar_100_test[b'data'],(np.shape(cifar_100_test[b'data'])[0],32,32,3))/255

    

    #Get rid if the first colum of attributes (id)
    attributes=attributes[:,2:(np.shape(attributes)[1])]


    regularization_const = .01
    #Get the model
    test_images_classification(cifar_100_train_images, cifar_100_train[b'fine_labels'], cifar_100_test_images, cifar_100_test[b'fine_labels'],regularization_const,classification_model_name,0)
    

def load_data_m_mnist():
    images = io.load_idx("Data/Morphomnist/train-images-idx3-ubyte.gz")
    labels = io.load_idx("Data/Morphomnist/train-labels-idx1-ubyte.gz")
    attributes=pd.read_csv("Data/Morphomnist/train-morpho.csv")
    print(np.shape(images),np.shape(labels),np.shape(attributes))
    return images/255,labels,pd.DataFrame.to_numpy(attributes)
def load_cifar():
    with open("Data/Cifar100/train", 'rb') as fo:
        dict_train = pickle.load(fo, encoding='bytes')
    with open("Data/Cifar100/test", 'rb') as fo:
        dict_test = pickle.load(fo, encoding='bytes')
    return dict_train,dict_test
if __name__ == '__main__':
    main()