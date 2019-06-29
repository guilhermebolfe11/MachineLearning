import math


class KNN(object):

    def euclideanDistance(x, y, length):
        distance = 0
        for i in range(length):
            distance += pow((x[i] - y[i]), 2)
        return math.sqrt(distance)
