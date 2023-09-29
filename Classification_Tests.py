import numpy as np
import tensorflow as tf
from Models import get_classifier_image

def test_images_classification(images_train, labels_train, images_test, labels_test,regularization_const, confidence_level,model_name,previous_model=0):

    #splitting the held out labels and images into calibration and test sets 50/50
    images_train, images_cal = np.split(images_train, [int(.9*len(images_train))])
    labels_train, labels_cal = np.split(labels_train, [int(.9*len(labels_train))])

    model = get_classifier_image(32,3,4,100,regularization_const)

    if(previous_model):
        model = model.load_weights(model_name)
    else:
        model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
        model.fit(images_train, labels_train, epochs=10, batch_size=32)
        #model.save_weights(model_name)

    #Getting the predictions of the calibration set

    predictions=model.predict(images_cal)

    #Making some edits to the base formula. Using the regression loss formula
    #Loss is defined as the excess number of classes until the right class is reached
    #This amkes the maximium error = number of classes-1

    #sorting the predictions
    argsort=tf.argsort(predictions,axis=1,direction='DESCENDING')
    prob_mass_values = []
    class_error_values = []
    #Getting the "regression" error and total probability mass of each prediction
    for next_prediction in range(len(predictions)):
        next_mass = 0
        class_count = 0

        for arg_value in argsort[next_prediction]:
            #If the right class is reached record the loss and probability mass
            if(arg_value == labels_cal[next_prediction]):
                prob_mass_values.append(next_mass+ predictions[next_prediction,arg_value])
                class_error_values.append(class_count)
                break
            else:
                next_mass += predictions[next_prediction,arg_value]
                class_count += 1

    average_loss = np.mean(class_error_values)

    #Get the CP bound for the calibration set

    #Get the corrected confidence value
    CP_index = np.ceil((len(labels_cal)+1)*(confidence_level))

    sorted_cp_values = np.sort(prob_mass_values, kind = "mergesort")
    CP_bound = sorted_cp_values[int(CP_index)]

    #Get the Hoeffding bound for the calibration set
    Hoeffding_bound = average_loss+99*np.sqrt((1/(2*len(labels_cal)))*np.log(2/(1-confidence_level)))

    #Running test set
    predictions=model.predict(images_test)

    #Testing validity and tightness of both bounds
    #CP bound is defiend as the amount of probability mass accumalted until >= CP bound
    #Hoeffding bound is the total mass outside of the top class

    argsort=tf.argsort(predictions,axis=1,direction='DESCENDING')


    prob_mass_values = []
    class_error_values = []

    #Getting the "regression" error and total probability mass of each prediction for the test set
    for next_prediction in range(len(predictions)):
        next_mass = 0
        class_count = 0

        for arg_value in argsort[next_prediction]:
            #If the right class is reached record the loss and probability mass
            if(arg_value == labels_test[next_prediction]):
                prob_mass_values.append(next_mass+ predictions[next_prediction,arg_value])
                class_error_values.append(class_count)
                break
            else:
                next_mass += predictions[next_prediction,arg_value]
                class_count += 1
    correct_cp = 0
    correct_hoeffding = 0

    excess_cp = []
    excess_hoeffding = []

    for i in range(len(prob_mass_values)):
        #Checking for correctness of the bounds
        if(CP_bound >= prob_mass_values[i]):
            correct_cp += 1
        if(Hoeffding_bound >= class_error_values[i]):
            correct_hoeffding += 1
        
        excess_cp.append(CP_bound-prob_mass_values[i])
        excess_hoeffding.append(Hoeffding_bound-class_error_values[i])

    CP_correct_ratio = correct_cp/len(prob_mass_values)
    Hoeffding_correct_ratio = correct_hoeffding/len(prob_mass_values)

    sorted_CP = np.sort(excess_cp)
    sorted_Hoeffding = np.sort(excess_hoeffding)

    excess_cp = sorted_CP[int((1-confidence_level)*len(prob_mass_values))]
    excess_hoeffding = sorted_Hoeffding[int((1-confidence_level)*len(prob_mass_values))]

    print("Number of correct classes for CP bound: ", CP_correct_ratio)
    print("Excess probability mass for CP bound: ", excess_cp)
    print("Number of correct classes for Hoeffding bound: ", Hoeffding_correct_ratio)
    print("Excess number of predicted classes for Hoeffding bound: ", excess_hoeffding)

            

