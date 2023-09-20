import tensorflow as tf

#Model to perform classification fo Mnist digits

def get_classifier_image(number_of_res_blocks, regularization_const=0.01):
    #Function whcih returns a model to perform classification of Mnist digits
    input = tf.keras.Input(shape=(28,28,1))
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', kernel_regularizer=tf.keras.regularizers.l2(regularization_const))(input)

    for _ in range(number_of_res_blocks):
        x = residual_block(x, 32, 5, regularization_const)
    
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=5, padding='same', kernel_regularizer=tf.keras.regularizers.l2(regularization_const))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(regularization_const))(x)

    
    model = tf.keras.Model(inputs=input, outputs=x)

    return model

def residual_block(x, filters, kernel_size, regularization_const):
    #basic residual block with skip connections, batch normalization, and l2 regularization
    y = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', kernel_regularizer=tf.keras.regularizers.l2(regularization_const))(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.LeakyReLU()(y)
    y = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', kernel_regularizer=tf.keras.regularizers.l2(regularization_const))(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.LeakyReLU()(y)
    y = tf.keras.layers.Add()([x, y])
    return y
