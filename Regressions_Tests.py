import tensorflow as tf
import numpy as np
from Models import get_regressor_image_quantile, get_regressor_image_base
def test_images_regression(images_train, labels_train, images_test, labels_test,regularization_const, confidence_level,model_name,previous_model=0):

    #splitting the held out labels and images into calibration and test sets 50/50
    images_train, images_cal = np.split(images_train, [int(.9*len(images_train))])
    labels_train, labels_cal = np.split(labels_train, [int(.9*len(labels_train))])

    if(previous_model):
        model = model.load_weights(model_name)
    else:

        model_max_loss = get_regressor_image_base(32,3,4,1,regularization_const)

        model_max_loss.compile(optimizer='adam',
                        loss=negative_loss)
        
        model_max_loss.fit(images_train, labels_train, epochs=10, batch_size=32)

        model_quantile = get_regressor_image_quantile(32,3,4,1,regularization_const)

        model_quantile.compile(optimizer='adam',
                        loss=pinball_loss_95)

        model_base = get_regressor_image_base(32,3,4,1,regularization_const)
        
        #Uses l1 loss
        model_base.compile(optimizer='adam',
                           loss = 'mae')
    
    max_mse = model_max_loss.predict(images_cal)

    #Getting the predictions of the calibration set

    predictions_max=model_max_loss.predict(images_cal)

    predictions = model.predict(images_cal)

    #calculate MSE

    Max_l1 = np.mean(np.abs(predictions_max-labels_test))

    l1_norm = np.mean(np.abs(predictions-labels_test))

    signed_l1 = predictions-labels_test

    arg_sorted = np.argsort(signed_l1)

    #Seperating into above and below




    
def negative_loss(y_true, y_pred):
    loss = -1*tf.losses.mean_squared_error()(y_true, y_pred)
    
def pinball_loss_95(y_true, y_pred):
    tau = 0.95
    error = y_true - y_pred
    return tf.reduce_mean(tf.maximum(tau * error, (tau - 1) * error), axis=-1)

def pinball_loss_05(y_true, y_pred):
    tau = 0.05
    error = y_true - y_pred
    return tf.reduce_mean(tf.maximum(tau * error, (tau - 1) * error), axis=-1)