import tensorflow as tf
import numpy as np
from Models import get_regressor_image_quantile, get_regressor_image_base

def test_images_regression(images_train, labels_train, images_test, labels_test,regularization_const, confidence_level,model_name,previous_model=0):
    
    #splitting the held out labels and images into calibration and test sets
    images_train, images_cal = np.split(images_train, [int(.9*len(images_train))])
    labels_train, labels_cal = np.split(labels_train, [int(.9*len(labels_train))])


    model_max_loss = get_regressor_image_base(28,1,6,6,regularization_const)

    model_quantile = get_regressor_image_quantile(28,1,4,6,regularization_const)

    model_base = get_regressor_image_base(28,1,6,6,regularization_const)

    model_quantile.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss={"High": pinball_loss(.5+confidence_level/2), "Low": pinball_loss(.5-confidence_level/2)})
    
    model_max_loss.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=negative_loss(),
                metrics = ['mean_absolute_error'])
    
    #Uses l1 loss
    difference = 10
    previous_mae=0
    current_mae = 0
    epoch_count = 0
    
    while(difference>1 and epoch_count<100):
        history = model_max_loss.fit(images_train, labels_train, epochs=1, batch_size=32,verbose=0)

        previous_mae = current_mae
        current_mae = history.history['mean_absolute_error'][0]

        difference = np.abs(current_mae-previous_mae)
        epoch_count+=1

    model_base.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss = 'mae',
                    metrics = ['mean_absolute_error'])
    
    model_quantile.fit(images_train, {"High": labels_train, "Low": labels_train}, epochs=int(1.2*epoch_count), batch_size=32,verbose=0)

    
    model_base.fit(images_train, labels_train, epochs=int(1.2*epoch_count), batch_size=32,verbose=0)

    #Getting the predictions of the calibration set

    predictions_max=model_max_loss.predict(images_cal,verbose=0)

    predictions_base = model_base.predict(images_cal,verbose=0)

    
    predictions_high, predictions_low = model_quantile.predict(images_cal,verbose=0)

    #calculate L1 norm for Hoeffding bound

    Max_l1 = np.mean(np.abs(predictions_max-labels_cal),axis=0)

    l1_norm = np.mean(np.abs(predictions_base-labels_cal),axis=0)

    
    #Get the Hoeffding bound for the calibration set
    Hoeffding_bound = np.array(l1_norm+Max_l1*np.sqrt((1/(2*len(labels_cal)))*np.log(2/(1-confidence_level))))

    class_bounds = []
    
    CP_index = np.ceil((len(labels_cal)+1)*(confidence_level))
    
    for i in range(len(labels_cal[0])):

        next_labels = labels_cal[:,i]
        next_predictions_high = predictions_high[:,i]
        next_predictions_low = predictions_low[:,i]

        #Dealing with CP bound

        # dividing the predictions into high, low and middle
        high_indices = next_labels > next_predictions_high
        low_indices = next_labels < next_predictions_low
        middle_indices = np.logical_and(next_labels <= next_predictions_high,next_labels>= next_predictions_low)

        #calculating l1 distance for each of the three groups
        high_l1 = next_labels[high_indices] - next_predictions_high[high_indices]
        low_l1 = next_predictions_low[low_indices] - next_labels[low_indices]
        middle_l1 = np.maximum(next_labels[middle_indices] - next_predictions_high[middle_indices], next_predictions_low[middle_indices] - next_labels[middle_indices])

        #concat tf.tensors
        l1 = tf.concat([high_l1, low_l1, middle_l1], axis=0)

        #sorting, low to high
        l1 = tf.sort(l1)


        class_bounds.append(l1[int(CP_index)])

    #convert to tf.tensor
    class_bounds = np.array(class_bounds)
    #Now doing test set
    
    predictions_base = np.array(model_base.predict(images_test,verbose=0))

    
    predictions_high, predictions_low = model_quantile.predict(images_test,verbose=0)

    predictions_high = np.array(predictions_high)
    predictions_low = np.array(predictions_low)

    Hoeffing_upper = predictions_base + Hoeffding_bound
    Hoeffing_lower = predictions_base - Hoeffding_bound

    #Dealing with CP bound
    CP_high = predictions_high + class_bounds
    CP_low = predictions_low - class_bounds

    #calculating the number of predictions that are outside of the bounds
    correct_percent_hoef = []
    bound_hoef = []

    correct_percent_cp = []
    bound_cp = []

    for i in range(len(labels_test[0])):
        next_labels = labels_test[:,i]

        next_predictions_high_cp = CP_high[:,i]
        next_predictions_low_cp = CP_low[:,i]

        next_predictions_high_hoef = Hoeffing_upper[:,i]
        next_predictions_low_hoef = Hoeffing_lower[:,i]

        # dividing the predictions into high, low and middle for cp
        high_indices = next_labels > next_predictions_high_cp
        low_indices = next_labels < next_predictions_low_cp
        middle_indices = np.logical_and(next_labels <= next_predictions_high_cp,next_labels>= next_predictions_low_cp)

        next_cp_percent = np.sum(middle_indices)/len(labels_test)

        #calculating l1 distance for each of the three groups
        high_l1 = next_labels[high_indices] - next_predictions_high_cp[high_indices]
        low_l1 = next_predictions_low_cp[low_indices] - next_labels[low_indices]
        middle_l1 = np.maximum(next_labels[middle_indices] - next_predictions_high_cp[middle_indices], next_predictions_low_cp[middle_indices] - next_labels[middle_indices])

        #concat tf.tensors
        l1 = tf.concat([high_l1, low_l1, middle_l1], axis=0)

        #sorting, low to high
        l1 = tf.sort(l1)

        bound_cp.append(l1[int(CP_index)-1])
        correct_percent_cp.append(next_cp_percent)

        # dividing the predictions into high, low and middle for hoeffding
        high_indices = next_labels > next_predictions_high_hoef
        low_indices = next_labels < next_predictions_low_hoef
        middle_indices = np.logical_and(next_labels <= next_predictions_high_hoef,next_labels>= next_predictions_low_hoef)

        next_hoef_percent = np.sum(middle_indices)/len(labels_test)

        #calculating l1 distance for each of the three groups
        high_l1 = next_labels[high_indices] - next_predictions_high_hoef[high_indices]
        low_l1 = next_predictions_low_hoef[low_indices] - next_labels[low_indices]
        middle_l1 = np.maximum(next_labels[middle_indices] - next_predictions_high_hoef[middle_indices], next_predictions_low_hoef[middle_indices] - next_labels[middle_indices])

        #concat tf.tensors
        l1 = tf.concat([high_l1, low_l1, middle_l1], axis=0)

        #sorting, low to high
        l1 = tf.sort(l1)

        bound_hoef.append(l1[int(CP_index)-1])
        correct_percent_hoef.append(next_hoef_percent)

    
    print("Correct Hoeffding Percentages: ", correct_percent_hoef, "\nHoeffding Bounds: ", np.array(bound_hoef))
    print("Correct CP Percentages: ", correct_percent_cp, "\nCP Bounds: ", np.array(bound_cp))
    
    model_base.save(model_name+"_base")
    model_max_loss.save(model_name+"_max")
    model_quantile.save(model_name+"_quantile")

    

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