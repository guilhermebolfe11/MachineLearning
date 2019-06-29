from distancing.KNN import KNN

data1 = [0, 1, 'a']
data2 = [0, 0, 'b']
distance = KNN.euclideanDistance(data1, data2, 2)

print('Distance: ' + repr(distance))
