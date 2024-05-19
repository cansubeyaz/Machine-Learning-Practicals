## TASK 2.1
list_1 = [1, "this", "that", 3.14159]
#print((list_1[0])) ## first index
#print((list_1[-2])) ## second last element index

list_1.append(1212)
#print(list_1) ## add 1212 to the end of the list

list_1.pop()
## list_1.remove(1212)
#print(list_1) ## remove the final element in the list

#del list_1[0]
list_1.remove(1)
#print(list_1) ## remove the first element in the list

list_1[2] = 5
#print(list_1) ## change any element in a list to something else

#for i in list_1:
    #print(i)

## TASK 2.2

dict = {1:"one", "two":2, "float":3.14159, "int":1212, "this":"that"}
#print(dict.keys())
#print(dict.values())
dict["cansu"] = "beyaz"
#print(dict)

dict.pop(1)
#print(dict)

#for i,k in dict.items():
    #print(i,k)

## TASK 3

import numpy as np

vector = np.array([12, 1, 42, 3.14])
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

print(vector[0])
print(matrix[1,3])
print(vector**2)
print(matrix*2)

print("Transpoze :", vector.reshape(-1,1))
print("Transpoze :", matrix.reshape(-1,1))

random_matrix = np.random.randint(0,11, size=(3,4))
print(random_matrix)

def calculate(radius=5, pi=3.14):
    area = pi * radius**2
    circum = 2*pi*radius
    return area, circum
