import numpy as np
import tensorflow as tf
from morphomnist import io, morpho, perturb
import pandas as pd
import pickle
import math
from scipy.stats.distributions import chi2
from scipy.special import erfinv

def main():
    regression_model_name = "Models/MorphoMnist_Regression_Model"
    synthetic_model_name = ["Models/Synthetic_Model_Periodic", "Models/Weierstrass"]

    #Load the data
    images,label,attributes=load_data_m_mnist()

    morpho_train = images[0:int(.9*len(images))]
    morpho_test = images[int(.9*len(images)):len(images)]

    morpho_train_labels = label[0:int(.9*len(images))]
    morpho_test_labels = label[int(.9*len(images)):len(images)]

    morpho_train_attributes = attributes[0:int(.9*len(images)),1:]
    morpho_test_attributes = attributes[int(.9*len(images)):len(images),1:]

    regularization_const = 10**(-1.75)

    test_images_regression(morpho_train, morpho_train_attributes, morpho_test, morpho_test_attributes, regularization_const, .9,regression_model_name,previous_model=0)
    synthetic_tests(synthetic_function1,regularization_const, .9, 10000, -10, 10, 17, synthetic_model_name[0])
    synthetic_tests(synthetic_function2,regularization_const, .9, 10000, -10, 10, 17, synthetic_model_name[1])


def test_images_regression(images_train, labels_train, images_test, labels_test,regularization_const, confidence_level,model_name,previous_model=0):

    #splitting the held out labels and images into calibration and test sets
    images_train, images_cal = np.split(images_train, [int(.9*len(images_train))])
    labels_train, labels_cal = np.split(labels_train, [int(.9*len(labels_train))])

    model_base = tf.keras.models.load_model(model_name+"_base", compile=False)

    model_max_loss = tf.keras.models.load_model(model_name+"_max", compile=False)

    model_quantile = tf.keras.models.load_model(model_name+"_quantile", compile=False)
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

    print("Correctness Results for MorphMnist")
    print("Correct CP Percentages: ", correct_percent_cp, "\nCP Bounds: ", np.array(bound_cp),"\n")
    print("Correct Hoeffding Percentages: ", correct_percent_hoef, "\nHoeffding Bounds: ", np.array(bound_hoef),"\n\n")
    

def synthetic_tests(syn_function, regularization_const, confidence_level, number_of_points, domain_low, domain_high, polynomial_degree, model_name):

    #Getting networks
    model_base = tf.keras.models.load_model(model_name+"_base", compile=False)
    model_negative_loss = tf.keras.models.load_model(model_name+"_max", compile=False)
    model_quantile = tf.keras.models.load_model(model_name+"_quantile", compile=False)


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

    with open(model_name+"_bayes", 'rb') as f:
        reg = pickle.load(f)
        


    #Getting Calibration predictions
    predictions_base = model_base.predict(calibration_set_domain, verbose=0)
    predictions_max = model_negative_loss.predict(calibration_set_domain, verbose=0)
    predictions_low, predictions_high = model_quantile.predict(calibration_set_domain, verbose=0)

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

    predictions_base = model_base.predict(test_set_domain, verbose=0)
    predictions_base = np.array(predictions_base)[:,0]

    
    predictions_low, predictions_high = model_quantile.predict(test_set_domain, verbose=0)

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

    Bayes_min=np.sum((weights_min*np.transpose(test_set_domain_linear)),axis=0)
    average_values = np.sum((np.expand_dims(reg.coef_,axis=1)*np.ones(weights_offset.shape)*np.transpose(test_set_domain_linear)),axis=0)
    Bayes_max=np.sum((weights_max*np.transpose(test_set_domain_linear)),axis=0)


    #Can evaluate probalities directly given that the noise is Gaussian
    error_function = np.vectorize(erf)
    #Hoeffding
    Y_mean = syn_function(test_set_domain)
    
    Hoeffding_prob_correct = error_function(Hoeffing_upper,Y_mean,np.ones(number_of_points))-\
                            error_function(Hoeffing_lower,Y_mean,np.ones(number_of_points))
    #CP
    CP_prob_correct = error_function(CP_high,Y_mean,np.ones(number_of_points))-\
                            error_function(CP_low,Y_mean,np.ones(number_of_points))
    
    Bayes_prob_correct= error_function(Bayes_max,Y_mean,np.ones(number_of_points))-\
                            error_function(Bayes_min,Y_mean,np.ones(number_of_points))
    
    True_bound_upper = Y_mean + 2**(1/2)*erfinv(.5+confidence_level/2)
    True_bound_lower = Y_mean + 2**(1/2)*erfinv(-(.5+confidence_level/2))

    print("Correctness Results for Synthetic Set")
    print("CP Coverage: ", np.mean(CP_prob_correct))
    print("CP Lower Bound Precision: ", np.mean(CP_low-True_bound_lower))
    print("CP Upper Bound Precision: ", np.mean(True_bound_upper-CP_high),"\n")
    print("Hoeffding Coverage: ", np.mean(Hoeffding_prob_correct))
    print("Hoeffding Lower Bound Precision: ", np.mean(Hoeffing_lower-True_bound_lower))
    print("Hoeffding Upper Bound Precision: ", np.mean(True_bound_upper-Hoeffing_upper),"\n")
    print("Bayes Coverage: ", np.mean(Bayes_prob_correct))
    print("Bayes Lower Bound Precision: ", np.mean(Bayes_min-True_bound_lower))
    print("Bayes Upper Bound Precision: ", np.mean(True_bound_upper-Bayes_max),"\n\n")

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
    
def synthetic_function1(x):
    return np.sin(x)*np.log(np.abs(x)+.01)

def synthetic_function2(x):
    first_term = np.ones(np.shape(x))

    for i in range(100):
        first_term += .5**i*np.cos(10**i*np.pi*x/100)
    return first_term

def load_data_m_mnist():
    images = io.load_idx("Data/Morphomnist/train-images-idx3-ubyte.gz")
    labels = io.load_idx("Data/Morphomnist/train-labels-idx1-ubyte.gz")
    attributes=pd.read_csv("Data/Morphomnist/train-morpho.csv")
    return images/255,labels,pd.DataFrame.to_numpy(attributes)
if __name__ == '__main__':
    tf.random.set_seed(25)
    np.random.seed(25)
    main()