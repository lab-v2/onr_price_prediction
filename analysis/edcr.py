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

