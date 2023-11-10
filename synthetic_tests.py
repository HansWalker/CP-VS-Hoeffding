import tensorflow as tf
import numpy as np
from Models import synthetic_network, synthetic_network_quantile
import math

def synthetic_tests(function, confidence_level, number_of_points, domain_low, domain_high):

    #Creating Dataset
    train_set_domain = np.random.uniform(domain_low, domain_high, number_of_points)
    train_set = function(train_set_domain)+np.random.normal(0,1,len(number_of_points))
    calibration_set_domain = np.random.uniform(domain_low, domain_high, number_of_points)
    calibration_set = function(calibration_set_domain)+np.random.normal(0,1,len(number_of_points))
    test_set_domain = np.random.uniform(domain_low, domain_high, number_of_points)
    test_set = function(test_set_domain)+np.random.normal(0,1,len(number_of_points))

    #Getting networks
    model_base = synthetic_network()
    model_negative_loss = synthetic_network()
    model_quantile = synthetic_network_quantile()

    #Compiling models
    model_base.compile(optimizer='adam',
                        loss = 'mae')
    model_negative_loss.compile(optimizer='adam',
                        loss=negative_loss)
    model_quantile.compile(optimizer='adam',
                        loss={"Low": pinball_loss_05,"High": pinball_loss_95})
    
    #Fitting models
    model_base.fit(train_set_domain, train_set, epochs=10, batch_size=32)
    model_negative_loss.fit(train_set_domain, train_set, epochs=10, batch_size=32)
    model_quantile.fit(train_set_domain, {"Low": train_set, "High": train_set}, epochs=10, batch_size=32)

    #Getting Calibration predictions
    predictions_base = model_base.predict(calibration_set_domain)
    predictions_max = model_negative_loss.predict(calibration_set_domain)
    predictions_low, predictions_high = model_quantile.predict(calibration_set_domain)

    predictions_high = np.array(predictions_high)
    predictions_low = np.array(predictions_low)

    #Hoeffding Bound
    Max_l1 = np.mean(np.abs(predictions_max-calibration_set),axis=0)

    l1_norm = np.mean(np.abs(predictions_base-calibration_set),axis=0)

    Hoeffding_bound = np.array(l1_norm+Max_l1*np.sqrt((1/(2*number_of_points))*np.log(2/(1-confidence_level))))
    
    CP_index = np.ceil((number_of_points+1)*(confidence_level))


    high_indices = calibration_set > predictions_high
    low_indices = calibration_set < predictions_low
    middle_indices = np.logical_and(calibration_set <= predictions_high,calibration_set>= predictions_low)

    #calculating l1 distance for each of the three groups
    high_l1 = calibration_set[high_indices] - predictions_high[high_indices]
    low_l1 = predictions_low[low_indices] - calibration_set[low_indices]
    middle_l1 = np.maximum(calibration_set[middle_indices] - predictions_high[middle_indices], predictions_low[middle_indices] - calibration_set[middle_indices])

    #concat tf.tensors
    l1 = tf.concat([high_l1, low_l1, middle_l1], axis=0)

    #sorting, low to high
    l1 = tf.sort(l1)


    CP_bound = l1[int(CP_index)]

    predictions_base = np.array(model_base.predict(test_set_domain))

    
    predictions_high, predictions_low = model_quantile.predict(test_set_domain)

    predictions_high = np.array(predictions_high)
    predictions_low = np.array(predictions_low)

    Hoeffing_upper = predictions_base + Hoeffding_bound
    Hoeffing_lower = predictions_base - Hoeffding_bound

    CP_high = predictions_high + CP_bound
    CP_low = predictions_low - CP_bound

    #Can evaluate probalities directly given that the noise is Gaussian
    error_function = np.vectorize(erf)
    #Hoeffding
    Y_mean = function(test_set_domain)
    
    Hoeffdiing_prob_correct = error_function(Hoeffing_upper,Y_mean,np.ones(number_of_points))-\
                            error_function(Hoeffing_lower,Y_mean,np.ones(number_of_points))
    #CP
    CP_prob_correct = error_function(CP_high,Y_mean,np.ones(number_of_points))-\
                            error_function(CP_low,Y_mean,np.ones(number_of_points))
    
    True_bound_upper = Y_mean + 1.9599639845400548
    True_bound_lower = Y_mean - 1.9599639845400548

    print("Hoeffding Correct: ", np.mean(Hoeffdiing_prob_correct))
    print("CP Correct: ", np.mean(CP_prob_correct))

def erf(x,mu,sig):
    return (1.0 + math.erf((x-mu) / (math.sqrt(2.0)*sig))) / 2.0

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
