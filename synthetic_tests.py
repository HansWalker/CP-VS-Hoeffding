import tensorflow as tf
import numpy as np
from Models import synthetic_network, synthetic_network_quantile
import math
from sklearn import linear_model
from scipy.stats.distributions import chi2
from scipy.special import erfinv
import pickle

def synthetic_tests(syn_function, regularization_const, confidence_level, number_of_points, domain_low, domain_high, polynomial_degree, model_name):

    #Creating Dataset
    train_set_domain = np.random.uniform(domain_low, domain_high, number_of_points)

    train_set = syn_function(train_set_domain)+np.random.normal(0,1,number_of_points)
    calibration_set_domain = np.random.uniform(domain_low, domain_high, number_of_points)
    calibration_set = syn_function(calibration_set_domain)+np.random.normal(0,1,number_of_points)
    test_set_domain = np.random.uniform(domain_low, domain_high, number_of_points)
    test_set = syn_function(test_set_domain)+np.random.normal(0,1,number_of_points)

    #Feature expansion for baysein ridge regression
    train_set_domain_linear = np.ones([number_of_points,1])
    next_domain = train_set_domain
    train_set_domain_linear = np.append(train_set_domain_linear,np.expand_dims(next_domain,axis=1),axis=1)
    for i in range(polynomial_degree-2):
        next_domain=next_domain*train_set_domain
        train_set_domain_linear = np.append(train_set_domain_linear,np.expand_dims(next_domain,axis=1),axis=1)

    test_set_domain_linear = np.ones([number_of_points,1])
    next_domain = test_set_domain
    test_set_domain_linear = np.append(test_set_domain_linear,np.expand_dims(next_domain,axis=1),axis=1)
    for i in range(polynomial_degree-2):
        next_domain=next_domain*test_set_domain
        test_set_domain_linear = np.append(test_set_domain_linear,np.expand_dims(next_domain,axis=1),axis=1)

    #Getting networks
    model_base = synthetic_network(regularization_const)
    model_negative_loss = synthetic_network(regularization_const)
    model_quantile = synthetic_network_quantile(regularization_const)
    reg = linear_model.BayesianRidge()

    #Compiling models
    model_base.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss = 'mae',
                    metrics = ['mean_absolute_error'])
    model_negative_loss.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                loss=negative_loss(),
                                metrics = ['mean_absolute_error'])
    model_quantile.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss={"Low": pinball_loss(.5-confidence_level/2),
                          "High": pinball_loss(.5+confidence_level/2)})
    
    #Fitting models
    difference = 10
    previous_mae=0
    current_mae = 0
    epoch_count = 0
    
    while(difference>1 and epoch_count<100):
        history = model_negative_loss.fit(train_set_domain, train_set, epochs=1, batch_size=32,verbose=0)

        previous_mae = current_mae
        current_mae = history.history['mean_absolute_error'][0]

        difference = np.abs(current_mae-previous_mae)
        epoch_count+=1

    model_base.fit(train_set_domain, train_set, epochs=int(1.2*epoch_count), batch_size=32,verbose=0)
    model_quantile.fit(train_set_domain, {"Low": train_set, "High": train_set}, epochs=int(1.2*epoch_count), batch_size=32,verbose=0)
    reg.fit(train_set_domain_linear,train_set)

    #Getting Calibration predictions
    predictions_base = model_base.predict(calibration_set_domain,verbose=0)
    predictions_max = model_negative_loss.predict(calibration_set_domain,verbose=0)
    predictions_low, predictions_high = model_quantile.predict(calibration_set_domain,verbose=0)

    predictions_high = np.array(predictions_high)[:,0]
    predictions_low = np.array(predictions_low)[:,0]

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

    predictions_base = model_base.predict(test_set_domain,verbose=0)
    predictions_base = np.array(predictions_base)[:,0]

    
    predictions_low, predictions_high = model_quantile.predict(test_set_domain,verbose=0)

    predictions_high = np.array(predictions_high)[:,0]
    predictions_low = np.array(predictions_low)[:,0]

    Hoeffing_upper = predictions_base + Hoeffding_bound
    Hoeffing_lower = predictions_base - Hoeffding_bound

    CP_high = predictions_high + CP_bound
    CP_low = predictions_low - CP_bound

    cov_matrix=reg.sigma_
    degrees = chi2.ppf(confidence_level, df=polynomial_degree)

    inner_term = np.matmul(cov_matrix,np.transpose(test_set_domain_linear))
    constant_term = degrees/np.sum(inner_term*np.transpose(test_set_domain_linear),axis=0)
    weights_offset = (constant_term)*(np.matmul(cov_matrix,np.transpose(test_set_domain_linear)))

    weights_max = np.expand_dims(reg.coef_,axis=1)*np.ones(weights_offset.shape)+weights_offset
    weights_min = np.expand_dims(reg.coef_,axis=1)*np.ones(weights_offset.shape)-weights_offset

    min_values=np.sum((weights_min*np.transpose(test_set_domain_linear)),axis=0)
    average_values = np.sum((np.expand_dims(reg.coef_,axis=1)*np.ones(weights_offset.shape)*np.transpose(test_set_domain_linear)),axis=0)
    max_values=np.sum((weights_max*np.transpose(test_set_domain_linear)),axis=0)


    #Can evaluate probalities directly given that the noise is Gaussian
    error_function = np.vectorize(erf)
    #Hoeffding
    Y_mean = syn_function(test_set_domain)
    
    Hoeffding_prob_correct = error_function(Hoeffing_upper,Y_mean,np.ones(number_of_points))-\
                            error_function(Hoeffing_lower,Y_mean,np.ones(number_of_points))
    #CP
    CP_prob_correct = error_function(CP_high,Y_mean,np.ones(number_of_points))-\
                            error_function(CP_low,Y_mean,np.ones(number_of_points))
    
    Bayes_prob_correct= error_function(max_values,Y_mean,np.ones(number_of_points))-\
                            error_function(min_values,Y_mean,np.ones(number_of_points))
    
    True_bound_upper = Y_mean + 2**(1/2)*erfinv(.5+confidence_level/2)
    True_bound_lower = Y_mean + 2**(1/2)*erfinv(-(.5+confidence_level/2))

    print("Hoeffding Correct: ", np.mean(Hoeffding_prob_correct))
    print("CP Correct: ", np.mean(CP_prob_correct))
    print("Bayes Correct: ", np.mean(Bayes_prob_correct))

    model_base.save(model_name+"_base")
    model_negative_loss.save(model_name+"_max")
    model_quantile.save(model_name+"_quantile")

    with open(model_name+"_bayes",'wb') as f:
        pickle.dump(reg,f)

def erf(x,mu,sig):
    return (1.0 + math.erf((x-mu) / (math.sqrt(2.0)*sig))) / 2.0

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
