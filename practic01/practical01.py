## TASK 2.1
'''
# 1. Create a list with the given elements
my_list = [1, 'this', 'that', 3.14159]

# 2. Index into the first element and print it
print(my_list[0])

# 3. Print the second last element when the length of the list is unknown
print(my_list[-2])

# 4. Add 1212 to the end of the list
my_list.append(1212)

# 5. Remove the final element in the list
my_list.pop()

# 6. Remove the first element in the list
del my_list[0]

# 7. Change any element in the list to something else
# Let's change the second element to 'something'
my_list[1] = 'something'

# 8. Use a for loop to print out the values in the list
for item in my_list:
    print(item)

'''

## TASK 2.2
'''
# 1. Create a dictionary with the given key-value pairs
my_dict = {
    1: 'one',
    'two': 2,
    'float': 3.14159,
    'int': 1212,
    'this': 'that'
}

# 2. Use the key to print out the value of a dictionary
print(my_dict[1])

# 3. Add a key-value pair to the dictionary
my_dict['new_key'] = 'new_value'
print(my_dict)
# 4. Remove a key-value pair from the dictionary
del my_dict['float']
print(my_dict)
# 5. Use a for loop to print out the key-value pairs
for key, value in my_dict.items():
    print(key, value)

'''
## TASK 3)

import numpy as np

# 1. Create the numpy vector and matrix
vector = np.array([12, 1, 42, 3.14])
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

# 2. Index into the vector and matrix and print some values
print("Vector:", vector[0], vector[-1])
print("Matrix:", matrix[0, 0], matrix[1, 2])

# 3. Add, subtract, multiply, and divide both the vector and matrix by a constant
constant = 2
print("Vector operations:")
print("Addition:", vector + constant)
print("Subtraction:", vector - constant)
print("Multiplication:", vector * constant)
print("Division:", vector / constant)

print("Matrix operations:")
print("Addition:", matrix + constant)
print("Subtraction:", matrix - constant)
print("Multiplication:", matrix * constant)
print("Division:", matrix / constant)

# 4. Add, subtract, multiply, and divide the matrix with the vector
print("Matrix-vector operations:")
print("Addition:", matrix + vector)
print("Subtraction:", matrix - vector)
print("Multiplication:", matrix * vector)  # Element-wise multiplication
print("Division:", matrix / vector)  # Element-wise division

# Transpose the vector
vector_transposed = vector.reshape(-1, 1)  # Reshape to a column vector
print("Transposed vector:")
print(vector_transposed)

# 5. Create a random numpy matrix of size 3×4, with values in the range 0 → 10
random_matrix = np.random.randint(0, 11, size=(3, 4))
print("Random Matrix:")
print(random_matrix)

# Element-wise matrix operations with the new matrix
print("Element-wise Matrix operations:")
print("Addition:", matrix + random_matrix)
print("Subtraction:", matrix - random_matrix)
print("Multiplication:", matrix * random_matrix)
print("Division:", matrix / random_matrix)

# Dot product between two matrices
# We can only perform dot product if the number of columns in the first matrix equals the number of rows in the second matrix
# Transpose the second matrix to make the dot product possible
dot_product = np.dot(matrix, random_matrix.T)
print("Dot Product:")
print(dot_product)

## TASK 4)
'''
def calculate_circle_properties(radius, pi=3.14):
    diameter = 2 * radius
    circumference = 2 * pi * radius
    area = pi * radius ** 2
    return diameter, circumference, area

# Example usage:
radius = 5
diameter, circumference, area = calculate_circle_properties(radius)
print("For a circle with radius", radius)
print("Diameter:", diameter)
print("Circumference:", circumference)
print("Area:", area)
'''
## TASK 5)
'''
class Rectangle:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        if height == width:
            self.name = 'square'
        else:
            self.name = 'rectangle'

    def calculate_circumference(self):
        return 2 * (self.height + self.width)

    def calculate_area(self):
        return self.height * self.width

    def calculate_and_print(self):
        circumference = self.calculate_circumference()
        area = self.calculate_area()
        print(f"For the {self.name}:")
        print("Circumference:", circumference)
        print("Area:", area)

# Example usage:
rectangle1 = Rectangle(5, 5)  # square
rectangle2 = Rectangle(4, 6)  # rectangle

rectangle1.calculate_and_print()
rectangle2.calculate_and_print()
'''