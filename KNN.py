import util
import numpy as np

class KNN():
    """ K Nearest Neighbors Classifier 

    Attributes:

        k (int): How many neighbors to consider (default: 5)
        dist_func (Callable): Distance function (default: euclidean_distance)
        X (np.array): Input training data
        Y (np.array): Output (e.g., gold prediction) training data
    """
    def __init__(self, task, k:int=5, dist_func=util.euclidean_distance):
        self.task = task
        self.k = k
        self.dist_func = dist_func

    def fit(self, X, Y):
        """ Adds X and Y from train to class """ 
        self.X = X
        self.Y = Y

    def predict(self, X_test: np.array) -> np.array:
        """ Makes prediction based on k closest neighbors and classification task
            Assumes that fitting data is already inputted to self.X and self.Y
        Args:
            X_test (np.array): Input test data

        Returns:
            np.array: Predictions; labels if self.task is classification or values if self.task is regression
        """ 

        ##Find the indices of closest k points
        super_large_number = 1000000000000000000000000000
        assert(self.k < len(self.X))
        distances = util.euclidean_distance(X_test, self.X)
        important_indices = []
        minimum = super_large_number
        minIndex = -1
        while (len(important_indices) != self.k):
            for i in range(len(distances)):
                if i not in important_indices:
                    if distances[i] < minimum:
                        minimum = distances
                        minIndex = i
            important_indices.append(minIndex)
            minimum=super_large_number

        ##get the outputs associated with those indices
        value_list = []
        for i in range(len(important_indices)):
            value_list.append(self.Y[important_indices[i]])

        if (self.task == "Classification"):
            mode = util.mode(value_list)
            return mode
        elif (self.task == "Regression"):
            unweighted_avg = sum(value_list) / len(value_list)
            return unweighted_avg
        else:
            print("Self.task is not a valid task type. Please input Classification or Regression")
    


if __name__ == '__main__':

    x = np.array([[1,2,3], [2,0,1], [4,4,2], [3,2,0], [1,5,1]]) 
    y_labels = np.array(['A', 'C', 'B', 'A', 'B'])
    y_values = np.array([3, 5, -1, 2, 0])



    ## Write test cases using the toy data above (or you can create your own toy data!)

    ## classfication test
    my_KNN_classification = KNN("Classification", 3)
    my_KNN_classification.fit(x, y_labels)
    print("Classfication Prediction Example: ", my_KNN_classification.predict([2, 2, 4]))

    ## Regression test
    my_KNN_regression = KNN("Regression", 3)
    my_KNN_regression.fit(x, y_values)
    print("Regression Prediction Example: ", my_KNN_regression.predict([2, 2, 4]))

