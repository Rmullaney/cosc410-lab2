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
            important_indices.append(i)
            minimum=super_large_number
        class_list = []
        for i in range(len(important_indices)):
            class_list.append(self.Y[important_indices[i]])
        mode = util.mode(class_list)
        return mode


if __name__ == '__main__':

    x = np.array([[1,2,3], [2,0,1], [4,4,2], [3,2,0], [1,5,1]]) 
    y_labels = np.array(['A', 'C', 'B', 'A', 'B'])
    y_values = np.array([3, 5, -1, 2, 0])



    ## Write test cases using the toy data above (or you can create your own toy data!)
    my_KNN = KNN()