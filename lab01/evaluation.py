from typing import List

def precision_recall(expected_results: List[bool], actual_results: List[bool]) -> (float, float):
    """Compute the precision and recall of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The precision of the predicted results.
        float
            The recall of the predicted results.
    """
    ##counts TP FP FN
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(expected_results)):
        if expected_results[i] == True and actual_results[i] == True:
            TP += 1
        elif expected_results[i] == False and actual_results[i] == True:
            FP += 1
        elif expected_results[i] == True and actual_results[i] == False:
            FN += 1
    #calculates recall and precision        
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    
    return (precision, recall)
   

def F1_score(expected_results: List[bool], actual_results: List[bool]) -> float:
    """Compute the F1-score of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The F1-score of the predicted results.
    """
    precision, recall = precision_recall(expected_results, actual_results)
    
    F1 = 2*precision*recall/(precision+recall)
    
    return F1
    
