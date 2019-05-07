import numpy as np


class NeuralNetwork(object):
    def __init__(self, neu_input, neu_hidden, neu_output):
        self.neu_input = neu_input
        self.neu_output = neu_output
        self.neu_hidden = neu_hidden

        self.WI = np.random.randn(self.neu_input, self.neu_hidden)
        self.WO = np.random.randn(self.neu_hidden, self.neu_output)

        self.error = 0
        self.delta = 0
        self.hid_error = 0
        self.hid_delta = 0
        self.inp = 0
        self.hid = 0
        self.out = 0

    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def sigmoidDerivative(self, a):
        return a * (1 - a)

    def backPropagtion(self, X, y, result):
        self.error = y - result
        self.delta = self.error * self.sigmoidDerivative(result)

        self.hid_error = self.delta.dot(self.WO.T)
        self.hid_delta = self.hid_error * self.sigmoidDerivative(self.hid)

        self.WI += X.T.dot(self.hid_delta)
        self.WO += self.hid.T.dot(self.delta)

    def training(self, X, y, error):
        print("----------------- TRAINING ------------------")
        result = self.predict(X)
        self.backPropagtion(X, y, result)
        print("Error: ", np.mean(np.square(self.error)))
        while np.mean(np.square(self.error)) > error:
            self.backPropagtion(X, y, result)
            result = self.predict(X)

        print("Error: ", np.mean(np.square(self.error)))


    def predict(self, X):
        self.inp = np.dot(X, self.WI)

        self.hid = self.sigmoid(self.inp)

        self.out = np.dot(self.hid, self.WO)

        result = self.sigmoid(self.out)

        return result

    def test(self, x, y):
        print("------------------- TEST --------------------")
        print("------------------- INPUT -------------------")
        print("Input: \n" + str(x))
        print("------------------- EXPECTED OUTPUT ---------")
        print("Expected Output: \n" + str(y))
        print()
        data = self.predict(x)
        for i in data:
            for j in range(0, 4):
                if i[j] > 0.5:
                    i[j] = 1
                else:
                    i[j] = 0
        print("------------------- PREDICT OUTPUT ----------")
        print("Predict Output: \n" + str(data))
        print("------------------- ERROR -------------------")
        print("Error: \n" + str(np.mean(np.square(y - data))))
        print("------------------- END TEST ----------------")
        print("\n")

    def normalize(self, X, y):
        X = X / np.amax(X, axis=0)
        y = y / np.amax(y, axis=0)
        return X, y