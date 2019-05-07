import numpy as np

from NeuralNetwork import NeuralNetwork

print("----------------- NEURAL NETWORK -------------------")
print("Rede perceptron capaz de reconhecer os 10 digitos numericos\n"
      " a partir de uma matriz 3x5 pixels, como nos exemplos abaixo.\n"
      "XXX    X   XXX  XXX\n"
      "X X   XX     X    X\n"
      "X X    X   XXX   XX\n"
      "X X    X   X      X\n"
      "XXX   XXX  XXX  XXX\n")

print("----------------- LOAD FILES TRAINING --------------")
file = input("File: ")
X = np.loadtxt(file, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14))
y = np.loadtxt(file, usecols=(15, 16, 17, 18))

print("----------------- LOAD FILES TRAINING --------------")
hidden_size = int(input("Hidden Size: "))
Net = NeuralNetwork(15, hidden_size, 4)

print("----------------- MAXIMUM ERROR ---------------------")
error = float(input("Error: "))
Net.training(X, y, error)

ok = 1
while (ok == 1):
    print("----------------- LOAD FILES TEST ---------------")
    file_test = input("File Test: ")
    X1 = np.loadtxt(file_test, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14))
    y1 = np.loadtxt(file_test, usecols=(15, 16, 17, 18))

    Net.test(X1, y1)

    print("----------------- NEW TEST ----------------------")
    ok = int(input("Perform New Test?"))

print("----------------- END NEURAL NETWORK ----------------")
