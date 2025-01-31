import doctest
import numpy as np
import math


def mse(pred:np.array, true:np.array) -> float:
    """
    Input:
        pred: numpy array of predicted values
        true: numpy array of true values
    Output: 
        Mean squared error as float; rounded to two decimal points

    >>> mse(np.array([3,2,1]), np.array([3,1,2]))
    0.67
    
    >>> mse(np.array([10, 15, -4, 0]), np.array([0, 15, 4, 0]))
    41.0

    """
    subsum=0
    for i in range(len(pred)):
        subsum += (pred[i] - true[i]) * (pred[i] - true[i])
    return float(round(subsum/len(pred),2))


def accuracy(pred:np.array, true:np.array) -> float:
    """
    Input:
        pred: numpy array of predicted labels
        true: numpy array of true labels
    Output:
        accuracy: float as percent; rounded to two decimal points

    >>> accuracy(np.array([1,1,1,1,1,0]), np.array([0,0,1,1,1,1]))
    50.0

    >>> accuracy(np.array(['A','A','B','A', 'A', 'C', 'C']), np.array(['C','A','B','A', 'A', 'C', 'B']))
    71.43
    """
    hits = 0
    for i in range(len(pred)):
        if pred[i] == true[i]:
            hits += 1
    accuracy = 100 * round(hits/len(pred), 4)
    return accuracy

def precision(pred:np.array, true:np.array) -> float:
    """
    Input:
        pred: numpy array of predicted labels
        true: numpy array of true labels
    Output:
        macro average precision: float as percent; rounded to two decimal points

    >>> precision(np.array([1,1,1,1,1,0]), np.array([0,0,1,1,1,1]))
    30.0

    >>> precision(np.array(['A','A','B','A', 'A', 'C', 'C']), np.array(['C','A','B','A', 'A', 'C', 'B']))
    75.0
    """
    dist_list = []
    for i in range(len(pred)):
        if pred[i] not in dist_list:
            dist_list.append(pred[i])
    subsum = 0
    for j in range(len(dist_list)):
        current_class = dist_list[j]
        occurrences = 0
        hits = 0
        for k in range(len(pred)):
            if (pred[k] == current_class):
                occurrences += 1
                if (true[k] == current_class):
                    hits +=1
        subsum += 100 * (hits/occurrences)
    return round(subsum/len(dist_list), 2)

def recall(pred:np.array, true:np.array) -> float:
    """
    Input:
        pred: numpy array of predicted labels
        true: numpy array of true labels
    Output:
        macro average recall: float as percent; rounded to two decimal points

    >>> recall(np.array([1,1,1,1,1,0]), np.array([0,0,1,1,1,1]))
    37.5

    >>> recall(np.array(['A','A','B','A', 'A', 'C', 'C']), np.array(['C','A','B','A', 'A', 'C', 'B']))
    66.67
    """
    dist_list = []
    for i in range(len(true)):
        if true[i] not in dist_list:
            dist_list.append(true[i])
    subsum = 0
    for j in range(len(dist_list)):
        current_class = dist_list[j]
        occurrences = 0
        hits = 0
        for k in range(len(true)):
            if (true[k] == current_class):
                occurrences += 1
                if (pred[k] == current_class):
                    hits +=1
        subsum += 100 * (hits/occurrences)
    return round(subsum/len(dist_list), 2)


def fscore(pred, true, f_val):
    """
    Input:
        pred: numpy array of predicted labels
        true: numpy array of true labels
    Output:
        macro average f-score: float; rounded to two decimal points

    >>> fscore(np.array(['A','A','B','A', 'A', 'C', 'C']), np.array(['C','A','B','A', 'A', 'C', 'B']), f_val=1)
    0.67

    >>> fscore(np.array(['A','A','B','A', 'A', 'C', 'C']), np.array(['C','A','B','A', 'A', 'C', 'B']), f_val=0.5)
    0.71

    >>> fscore(np.array([1,1,1,1,1,0]), np.array([0,0,1,1,1,1]), f_val=2)
    0.36
    """
    dist_list = []
    for i in range(len(true)):
        if true[i] not in dist_list:
            dist_list.append(true[i])
    subsum = 0
    f_val_squared = f_val * f_val
    for j in range(len(dist_list)):
        current_class = dist_list[j]
        occurrences_pred = 0
        occurrences_true = 0
        hits_given_pred = 0
        hits_given_true = 0
        for k in range(len(true)):
            if (true[k] == current_class):
                occurrences_true += 1
                if (pred[k] == current_class):
                    hits_given_true +=1
            if (pred[k] == current_class):
                occurrences_pred += 1
                if (true[k] == current_class):
                    hits_given_pred +=1

        prec = hits_given_pred/occurrences_pred
        rec = hits_given_true/occurrences_true
        if (prec == 0 and rec == 0):
            subsum = 0
        else:
            subsum += (((f_val_squared+1) * prec * rec) / ((f_val_squared * prec) + rec))


    return round(subsum/len(dist_list), 2)

def euclidean_distance(point: np.array, data: np.array) -> np.array:
    """ Calculates the euclidean distance for a point against all the data.

    Args:
        point (np.array): A single sample from test data
        data (np.array): All of the train data to calculate the distance for
    Returns:
        np.array: Distances between the point and all the data

    >>> euclidean_distance(np.array([0,0]), np.array([[3,3], [1,2], [2,1], [-3,-3]]))
    array([4.24264069, 2.23606798, 2.23606798, 4.24264069])
    
    >>> euclidean_distance(np.array([3,5,1]), np.array([[2,3,4], [3,5,2], [5,-1,0], [0,0,0]]))
    array([3.74165739, 1.        , 6.40312424, 5.91607978])
    """
    distance = []
    for i in range(len(data)):
        subsum = 0
        for j in range(len(point)):
            subsum += (point[j] - data[i][j]) * (point[j] - data[i][j])
        distance.append(round(math.sqrt(subsum), 8))
    return np.array(distance)


def mode(labels: np.array) -> list:
    """ Returns the mode value from a list 

    Args:
        labels (list): A list of labels
    Returns:
        List of values that are the mode. List should have one value if there is only one mode. 

    >>> mode(['cat', 'dog', 'cat', 'butterfly'])
    ['cat']

    >>> mode(['cat', 'dog', 'cat', 'butterfly', 'dog'])
    ['cat', 'dog']


    >>> mode([1, 0, 0, 1, 2, 3, 3, 3])
    [3]

    >>> mode([1, 0, 0, 1, 2, 3])
    [0, 1]
    """ 
    count = {}
    for i in range(len(labels)):
        if labels[i] in count:
            count[labels[i]] += 1
        else:
            count[labels[i]] = 1
    max = []
    maxNum = 0
    for key in count.keys():
        if count[key] > maxNum:
            max = [key]
            maxNum = count[key]
        elif count[key] == maxNum:
            if key not in max:
                max.append(key)
    max.sort()
    return max



if __name__ == '__main__':
    doctest.testmod(verbose=True)
    
