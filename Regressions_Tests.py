import tensorflow as tf
import numpy as np
from Models import get_regressor_image
def test_images_regression(images_train, labels_train, images_test, labels_test,regularization_const, confidence_level,model_name,previous_model=0):

    model = get_regressor_image(32,3,4,1,regularization_const)

    if(previous_model):
        model = model.load_weights(model_name)
    else:
        model.compile(optimizer='adam',
                        loss=tf.losses.mean_squared_error())
        model.fit(images_train, labels_train, epochs=10, batch_size=32)

        model_max_loss = get_regressor_image(32,3,4,1,regularization_const)

        model_max_loss.compile(optimizer='adam',
                        loss=negative_loss)
        model_max_loss.fit(images_train, labels_train, epochs=10, batch_size=32)
    
    max_mse = model_max_loss.predict(images_test)

    #Getting the predictions of the calibration set

    predictions_max=model_max_loss.predict(images_test)

    predictions = model.predict(images_test)

    #

    
def negative_loss(y_true, y_pred):
    loss = -1*tf.losses.mean_squared_error()(y_true, y_pred)