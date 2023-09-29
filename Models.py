import tensorflow as tf

#Model to perform classification fo Mnist digits

def get_classifier_image(input_shape,input_channels,number_of_res_blocks, output_dim, regularization_const=0.01):
    #Function whcih returns a model to perform classification of Mnist digits
    input = tf.keras.layers.Input(shape=(input_shape,input_shape,input_channels))
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=5, padding='same', kernel_regularizer=tf.keras.regularizers.l2(regularization_const))(input)

    for _ in range(number_of_res_blocks):
        x = residual_block(x, 64, 3, regularization_const)
    
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=5, padding='same', kernel_regularizer=tf.keras.regularizers.l2(regularization_const))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(output_dim, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(regularization_const))(x)
    

    
    model = tf.keras.Model(inputs=input, outputs=x)

    return model

def get_regressor_image(input_shape,input_channels,number_of_res_blocks, output_dim, regularization_const=0.01):
    #Function whcih returns a model to perform classification of Mnist digits
    input = tf.keras.layers.Input(shape=(input_shape,input_shape,input_channels))
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=5, padding='same', kernel_regularizer=tf.keras.regularizers.l2(regularization_const))(input)

    for _ in range(number_of_res_blocks):
        x = residual_block(x, 64, 3, regularization_const)
    
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=5, padding='same', kernel_regularizer=tf.keras.regularizers.l2(regularization_const))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(output_dim, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(regularization_const))(x)
    

    
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
