import tensorflow as tf
import numpy as np
from Models import get_regressor_image_quantile, get_regressor_image_base
def test_images_regression(images_train, labels_train, images_test, labels_test,regularization_const, confidence_level,model_name,previous_model=0):

    #splitting the held out labels and images into calibration and test sets 50/50
    images_train, images_cal = np.split(images_train, [int(.9*len(images_train))])
    labels_train, labels_cal = np.split(labels_train, [int(.9*len(labels_train))])


    model_max_loss = get_regressor_image_base(32,3,4,1,regularization_const)

    model_max_loss.compile(optimizer='adam',
                    loss=negative_loss)
    
    model_max_loss.fit(images_train, labels_train, epochs=10, batch_size=32)

    model_quantile = get_regressor_image_quantile(32,3,4,1,regularization_const)

    model_quantile.compile(optimizer='adam',
                    loss={"High": pinball_loss_95, "Low": pinball_loss_05})
    
    model_quantile.fit(images_train, {"High": labels_train, "Low": labels_train}, epochs=10, batch_size=32)

    model_base = get_regressor_image_base(32,3,4,1,regularization_const)
    
    #Uses l1 loss
    model_base.compile(optimizer='adam',
                        loss = 'mae')
    
    model_base.fit(images_train, labels_train, epochs=10, batch_size=32)

    #Getting the predictions of the calibration set

    predictions_max=model_max_loss.predict(images_cal)

    predictions_base = model_base.predict(images_cal)

    
    predictions_high, predictions_low = model_quantile.predict(images_cal)

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
        high_indices = next_labels > predictions_high
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
    
    predictions_base = np.array(model_base.predict(images_test))

    
    predictions_high, predictions_low = model_quantile.predict(images_test)

    predictions_high = np.array(predictions_high)
    predictions_low = np.array(predictions_low)

    Hoeffing_upper = predictions_base + Hoeffding_bound
    Hoeffing_lower = predictions_base - Hoeffding_bound

    #Dealing with CP bound
    CP_high = predictions_high + class_bounds
    CP_low = predictions_low - class_bounds

    #calculating the number of predictions that are outside of the bounds
    correct_percent_hoef = []
    correct_percent_cp = []

    for i in range(len(labels_test[0])):
        next_labels = labels_test[:,i]
        Hoeffding_correct = np.logical_and(next_labels <= Hoeffing_upper, next_labels>=Hoeffing_lower)
        CP_correct = np.logical_and(next_labels <= CP_high, next_labels>=CP_low)

        correct_percent_hoef.append(np.sum(Hoeffding_correct)/len(Hoeffding_correct))
        correct_percent_cp.append(np.sum(CP_correct)/len(CP_correct))
    
    print("Correct Hoeffding Percentages: ", correct_percent_hoef)
    print("Correct CP Percentages: ", correct_percent_cp)
    


    
def negative_loss(y_true, y_pred):
    loss = -1*tf.losses.mean_squared_error()(y_true, y_pred)
    return loss
    
def pinball_loss_95(y_true, y_pred):
    tau = 0.95
    error = y_true - y_pred
    return tf.reduce_mean(tf.maximum(tau * error, (tau - 1) * error), axis=-1)

def pinball_loss_05(y_true, y_pred):
    tau = 0.05
    error = y_true - y_pred
    return tf.reduce_mean(tf.maximum(tau * error, (tau - 1) * error), axis=-1)