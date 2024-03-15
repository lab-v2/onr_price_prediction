import numpy as np

# Support (s): fraction of samples in O where the body is true.
def calculate_support(y_pred, class_of_interest=1):
    return np.mean(y_pred == class_of_interest)

# Confidence (c): the number of times the body and head are true together divided by the number of times the body is true.
def calculate_confidence(y_pred, y_test, class_of_interest=1):
    true_positives = np.logical_and(y_pred == class_of_interest, y_test == class_of_interest).sum()
    predicted_positives = (y_pred == class_of_interest).sum()
    confidence = true_positives / predicted_positives if predicted_positives > 0 else 0
    return confidence

def calculate_precision(y_pred, y_test, class_of_interest=1):
    true_positives = np.logical_and(y_pred == class_of_interest, y_test == class_of_interest).sum()
    predicted_positives = (y_pred == class_of_interest).sum()
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    return precision

def calculate_recall(y_pred, y_test, class_of_interest=1):
    true_positives = np.logical_and(y_pred == class_of_interest, y_test == class_of_interest).sum()
    actual_positives = (y_test == class_of_interest).sum()
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    return recall


def apply_detection_and_correction(base_model_pred, correction_model_pred, y_test, epsilon_precision=0.5, epsilon_recall=0.5):
    corrected_preds = np.copy(base_model_pred)
    
    # Calculate precision and recall for the base model
    base_precision = calculate_precision(y_test, base_model_pred)
    base_recall = calculate_precision(y_test, base_model_pred)
    
    # Calculate support and confidence for the correction model
    correction_support = calculate_support(correction_model_pred)
    correction_confidence = calculate_confidence(correction_model_pred, y_test)

    # Calculate support and confidence for the base model
    base_support = calculate_support(base_model_pred)
    base_confidence = calculate_confidence(base_model_pred, y_test)
    
    # Error Detection based on conditions, need adjust??
    detection_condition = (correction_support < 1 - base_precision) and (correction_confidence >= epsilon_precision)
    
    # Apply correction if detection condition is met
    if detection_condition:
        # Find where base model is wrong
        potential_errors = base_model_pred != correction_model_pred
        
        # Apply corrections based on the correction model's predictions
        for i in range(len(base_model_pred)):
            if potential_errors[i]: # Apply correction only to potential errors
                corrected_preds[i] = correction_model_pred[i]
    
    return corrected_preds



