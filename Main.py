import numpy as np
import tensorflow as tf
from morphomnist import io, morpho, perturb
import pandas as pd
from Models import get_classifier_image
from Classification_Tests import test_images_classification
def main():
    #Load the data
    images,label,attributes=load_data()

    #Get rid if the first colum of attributes (id)
    attributes=attributes[:,2:(np.shape(attributes)[1])]

    #Split train images in train and calibration (80/20)
    images_train, images_heldout = np.split(images, [int(.8*len(images))])
    
    #Split of labels
    labels_train, labels_heldout = np.split(label, [int(.8*len(label))])
    #Split of attributes
    attributes_train, attributes_heldout = np.split(attributes, [int(.8*len(attributes))])


    regularization_const = .01
    #Get the model
    test_images_classification(images_train, labels_train, images_heldout, labels_heldout,regularization_const)
    

def load_data():
    images = io.load_idx("Data/train-images-idx3-ubyte.gz")
    labels = io.load_idx("Data/train-labels-idx1-ubyte.gz")
    attributes=pd.read_csv("Data/train-morpho.csv")
    print(np.shape(images),np.shape(labels),np.shape(attributes))
    return images/255,labels,pd.DataFrame.to_numpy(attributes)
if __name__ == '__main__':
    main()