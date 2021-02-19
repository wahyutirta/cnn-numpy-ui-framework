import numpy as np

array = []

for i in range(5):
    array = np.append(array,i)

array = np.array(array)
print(array)
print(np.average(array))

