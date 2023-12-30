import numpy as np
import tensorflow as tf
from morphomnist import io, morpho, perturb
import pandas as pd
import pickle
from Models import get_regressor_image_base
def main():

    #Load the data
    images,label,attributes=load_data_m_mnist()

    morpho_train = images[0:int(.9*len(images))]
    morpho_test = images[int(.9*len(images)):len(images)]

    morpho_train_labels = label[0:int(.9*len(images))]
    morpho_test_labels = label[int(.9*len(images)):len(images)]

    morpho_train_attributes = attributes[0:int(.9*len(images)),1:]
    morpho_test_attributes = attributes[int(.9*len(images)):len(images),1:]

    images_train=morpho_train
    labels_train=morpho_train_attributes
    images_test=morpho_test
    labels_test=morpho_test_attributes


    regularization_const_list1 =10**np.linspace(0,-1,5, endpoint = False)
    regularization_const_list2 =10**np.linspace(-1,-3, 16, endpoint = False)
    regularization_const_list3 =10**np.linspace(-3,-3.5,5)

    regularization_const_list = np.concatenate([regularization_const_list1,regularization_const_list2,regularization_const_list3])

    list_max = []
    list_base = []

    model_max_loss_prev = get_regressor_image_base(28,1,6,6,1).get_weights()
    model_base_prev = get_regressor_image_base(28,1,6,6,1).get_weights()

    for regularization_const in regularization_const_list:
        model_max_loss = get_regressor_image_base(28,1,6,6,regularization_const)

        model_max_loss.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        loss=negative_loss(),
                        metrics = ['mean_absolute_error'])
        
        #copy weights from previous model
        model_max_loss.set_weights(model_max_loss_prev)
        
        difference = 10
        previous_mae=0
        current_mae = 0
        epoch_count = 0 
        
        while(difference>1 and epoch_count<130):
            history = model_max_loss.fit(images_train, labels_train, epochs=1, batch_size=32,verbose=0)

            previous_mae = current_mae
            current_mae = history.history['mean_absolute_error'][0]

            difference = np.abs(current_mae-previous_mae)
            epoch_count+=1
        
        print("Mae: ",current_mae," Epochs: ",epoch_count," Regularization: ",regularization_const)
        model_base = get_regressor_image_base(28,1,6,6,regularization_const)


        #Uses l1 loss
        model_base.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        loss = 'mae',
                        metrics = ['mean_absolute_error'])
        
        #Copy weights from previous model
        model_base.set_weights(model_base_prev)

        history = model_base.fit(images_train, labels_train, epochs=int(1.2*epoch_count), batch_size=32,verbose=0)

        print("Mae: ",history.history['mean_absolute_error'][len(history.history['mean_absolute_error'])-1])

        predictions_max=model_max_loss.predict(images_train,verbose=0)

        predictions_base = model_base.predict(images_train,verbose=0)

        mae_max = np.mean(np.abs(predictions_max-labels_train))
        mae_base = np.mean(np.abs(predictions_base-labels_train))

        list_max.append(mae_max)
        list_base.append(mae_base)

        model_max_loss_prev = model_max_loss.get_weights()
        model_base_prev = model_base.get_weights()

        #save data arrays
        np.save("Results/Reg Constants", regularization_const_list)
        np.save("Results/Max list", list_max)
        np.save("Results/Base list", list_base)

def load_data_m_mnist():
    images = io.load_idx("Data/Morphomnist/train-images-idx3-ubyte.gz")
    labels = io.load_idx("Data/Morphomnist/train-labels-idx1-ubyte.gz")
    attributes=pd.read_csv("Data/Morphomnist/train-morpho.csv")
    return images/255,labels,pd.DataFrame.to_numpy(attributes)
def load_cifar():
    with open("Data/Cifar100/train", 'rb') as fo:
        dict_train = pickle.load(fo, encoding='bytes')
    with open("Data/Cifar100/test", 'rb') as fo:
        dict_test = pickle.load(fo, encoding='bytes')
    return dict_train,dict_test
class negative_loss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        self.mse = tf.keras.losses.MeanAbsoluteError()
    def call(self, y_true, y_pred):   
        loss = -1*(self.mse(y_true, y_pred))     
        return loss
      
class pinball_loss(tf.keras.losses.Loss):
    def __init__(self, quantile):
        super().__init__()
        self.tau = quantile
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(self.tau * error, (self.tau - 1) * error), axis=-1)
if __name__ == "__main__":
    tf.random.set_seed(25)
    np.random.seed(25)
    main()