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

def get_regressor_image_quantile(input_shape,input_channels,number_of_res_blocks, output_dim, regularization_const=0.01):
    #Function whcih returns a model to perform classification of Mnist digits
    input = tf.keras.layers.Input(shape=(input_shape,input_shape,input_channels))
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=5, padding='same', kernel_regularizer=tf.keras.regularizers.l2(regularization_const))(input)

    for _ in range(number_of_res_blocks):
        x = residual_block(x, 64, 3, regularization_const)
    
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=5, padding='same', kernel_regularizer=tf.keras.regularizers.l2(regularization_const))(x)
    x = tf.keras.layers.Flatten()(x)
    
    #Low quantile prediction
    x1 = tf.keras.layers.Dense(output_dim*10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization_const))(x)
    x1 = tf.keras.layers.Dense(output_dim*10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization_const))(x1)
    x1 = tf.keras.layers.Dense(output_dim*10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization_const))(x1)

    low_prediction = tf.keras.layers.Dense(output_dim, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(regularization_const), name="Low")(x1)

    #High quantile prediction
    x2 = tf.keras.layers.Dense(output_dim*10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization_const))(x)
    x2 = tf.keras.layers.Dense(output_dim*10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization_const))(x2)
    x2 = tf.keras.layers.Dense(output_dim*10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization_const))(x2)
    
    high_prediction = tf.keras.layers.Dense(output_dim, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(regularization_const), name="High")(x2)

    
    model = tf.keras.Model(inputs=input, outputs=[low_prediction, high_prediction])

    return model

def get_regressor_image_base(input_shape,input_channels,number_of_res_blocks, output_dim, regularization_const=0.01):
    #Function whcih returns a model to perform classification of Mnist digits
    input = tf.keras.layers.Input(shape=(input_shape,input_shape,input_channels))
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=5, padding='same', kernel_regularizer=tf.keras.regularizers.l2(regularization_const))(input)

    for _ in range(number_of_res_blocks):
        x = residual_block(x, 64, 3, regularization_const)
    
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=5, padding='same', kernel_regularizer=tf.keras.regularizers.l2(regularization_const))(x)
    x = tf.keras.layers.Flatten()(x)
    
    #Low quantile prediction
    x1 = tf.keras.layers.Dense(output_dim*10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization_const))(x)
    x1 = tf.keras.layers.Dense(output_dim*10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization_const))(x1)
    x1 = tf.keras.layers.Dense(output_dim*10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization_const))(x1)

    prediction = tf.keras.layers.Dense(output_dim, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(regularization_const), name="Low")(x1)

    
    model = tf.keras.Model(inputs=input, outputs=prediction)

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

def synthetic_network():
    #A network to approximate a 1 dimensional function
    inputs = tf.keras.layers.Input(shape=(1,))
    x = tf.keras.layers.Dense(100, activation='relu')(inputs)

    #A bunch of fully connect residual layers
    for i in range(10):
        x_1 = tf.keras.layers.Dense(100, activation='relu')(x)
        #Layer norm
        x = tf.keras.layers.LayerNormalization()(x_1)
        
        x = tf.keras.layers.Dense(100, activation='relu')(x)

        x = tf.keras.layers.LayerNormalization()(x)
        
        #Addition
        x = tf.keras.layers.Add()([x_1, x])
    x = tf.keras.layers.Dense(100, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='linear')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model

def synthetic_network_quantile():
    #A network to approximate a 1 dimensional function
    inputs = tf.keras.layers.Input(shape=(1,))
    x = tf.keras.layers.Dense(100, activation='relu')(inputs)

    #A bunch of fully connect residual layers
    for i in range(10):
        x_1 = tf.keras.layers.Dense(100, activation='relu')(x)
        #Layer norm
        x = tf.keras.layers.LayerNormalization()(x_1)
        
        x = tf.keras.layers.Dense(100, activation='relu')(x)

        x = tf.keras.layers.LayerNormalization()(x)
        
        #Addition
        x = tf.keras.layers.Add()([x_1, x])
    x_low=tf.keras.layers.Dense(100, activation='relu')(x)
    x_high=tf.keras.layers.Dense(100, activation='relu')(x)
    x_low = tf.keras.layers.Dense(1, activation='linear', name="Low")(x_low)
    x_high = tf.keras.layers.Dense(1, activation='linear', name="High")(x_high)

    model = tf.keras.Model(inputs=inputs, outputs=[x_low, x_high])

    return model
