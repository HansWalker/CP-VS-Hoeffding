import tensorflow as tf
import numpy as np
from Models import get_regressor_image
def test_images_regression(images_train, labels_train, images_test, labels_test,regularization_const, confidence_level,model_name,previous_model=0):

