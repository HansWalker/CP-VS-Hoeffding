import numpy as np
import tensorflow as tf
from Models import get_classifier_image

def test_images_classification(images_train, labels_train, images_heldout, labels_heldout,regularization_const, confidence_level=.95):

    #splitting the held out labels and images into calibration and test sets 50/50
    images_cal, images_test = np.split(images_heldout, [int(.5*len(images_heldout))])
    labels_cal, labels_test = np.split(labels_heldout, [int(.5*len(labels_heldout))])

    model = get_classifier_image(2,regularization_const)
    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    model.fit(images_train, labels_train, epochs=1)

    #Getting the predictions of the calibration set

    predictions=model.predict(images_cal)

    #Making some edits to the base formula. Using the regression loss formula
    #Loss is defined as the excess probability mass of the wrong classes predicted
    #until the right class is reached. The probability mass of the right class is
    #not counted

    #sorting the predictions
    argsort=tf.argsort(predictions,axis=1,direction='DESCENDING')
    loss_values = []
    prob_mass_values = []
    #Getting the "regression" error and total probability mass of each prediction
    for next_prediction in range(len(predictions)):
        next_mass = 0

        for arg_value in argsort[next_prediction]:
            #If the right class is reached record the loss and probability mass
            if(arg_value == labels_cal[next_prediction]):
                loss_values.append(next_mass)
                prob_mass_values.append(next_mass+predictions[next_prediction][arg_value])
            else:
                next_mass += predictions[next_prediction,arg_value]

    average_loss = np.mean(loss_values)

    #Get the CP bound for the calibration set

    #Get the corrected confidence value
    CP_index = np.ceil((len(labels_cal)+1)*(confidence_level))

    sorted_cp_values = np.sort(prob_mass_values, kind = "mergesort")
    CP_bound = sorted_cp_values[int(CP_index)]

    #Get the Hoeffding bound for the calibration set
    Hoeffding_bound = average_loss+np.sqrt((1/(2*len(labels_cal)))*np.log(2/(1-confidence_level)))

    #Running test set
    predictions=model.predict(images_test)

    #Testing validity and tightness of both bounds
    #CP bound is defiend as the amount of probability mass accumalted until >= CP bound
    #Hoeffding bound is the total mass outside of the top class

    argsort=tf.argsort(predictions,axis=1,direction='DESCENDING')
    correctness_CP = []

    correctness_Hoeffding = []

    #Getting the "regression" error and total probability mass of each prediction
    for next_prediction in range(len(predictions)):

        #Array is correctness, excess number of classes, excess probability mass
        found = False
        finished_CP = False
        next_cp = [0,0,0]

        finished_hoeffding = False
        next_hoeffding = [0,0,0]

        total_mass_cp = 0
        total_mass_hoeffding = -predictions[next_prediction,argsort[next_prediction,0]]
        for arg_value in argsort[next_prediction]:
            #If the right class is reached record the loss and probability mass
            if(finished_CP and finished_hoeffding):
                break
            
            #Recording excess number of classes beyond the found one
            if(found):

                #Updating the total mass
                total_mass_cp+=predictions[next_prediction,arg_value]
                total_mass_hoeffding+=predictions[next_prediction,arg_value]

                if(not finished_CP):
                    next_cp[2]+=predictions[next_prediction,arg_value]
                if(not finished_hoeffding):
                    next_hoeffding[2]+=predictions[next_prediction,arg_value]
            else:
                if(arg_value == labels_test[next_prediction]):
                    found = True
                else:
                    if(not finished_CP):
                        next_cp[1]+=1
                    if(not finished_hoeffding):
                        next_hoeffding[1]+=1
            if(total_mass_cp>=CP_bound):
                finished_CP = True
            if(total_mass_hoeffding>=Hoeffding_bound):
                finished_hoeffding = True
    cp_results = np.array(correctness_CP)

            

