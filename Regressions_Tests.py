import tensorflow as tf
import numpy as np
from Models import get_regressor_image
def test_images_regression(images_train, labels_train, images_test, labels_test,regularization_const, confidence_level,model_name,previous_model=0):

    model = get_regressor_image(32,3,4,1,regularization_const)

    if(previous_model):
        model = model.load_weights(model_name)
    else:
        model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
        model.fit(images_train, labels_train, epochs=10, batch_size=32)