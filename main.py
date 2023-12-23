import time

class MyRandom:
    def __init__(self, seed=time.time()):    # Initializer or constructor for the MyRandom class
        # Set the initial seed value for the random number generator
        self.seed = seed
        # Constants used for the generator algorithm
        self.a = 15
        self.b = 18
        self.c = 47

    def rand(self):    # Method to generate a pseudo-random number
        # Update the seed and ensure the result is an integer
        self.seed = int((self.a * self.seed + self.b) % self.c)
        # Return the new pseudo-random number
        return self.seed

    def random(self, *args):   # Method to generate a multi-dimensional array filled with random numbers
        # If no dimensions are provided, raise an error
        if len(args) == 0:
            raise ValueError("At least one dimension size is required")
        # Call the generate_array method with the provided dimensions
        return self.generate_array(args)

    def generate_array(self, dimensions, depth=0):    # Recursive method to generate an array of given dimensions
        # If the current depth equals the number of dimensions, return a random number
        if depth == len(dimensions):
            return self.rand()
        # Otherwise, build the next dimension recursively
        return [self.generate_array(dimensions, depth + 1) for _ in range(dimensions[depth])]

# Usage:
# rnd = MyRandom()  # Create an instance of MyRandom
# vector = rnd.random(3)  # Generate a 1D array (vector) with 3 random numbers
# matrix = rnd.random(3, 2)  # Generate a 2D array (matrix) with 3x2 random numbers
# tensor = rnd.random(3, 2, 4)  # Generate a 3D array (tensor) with 3x2x4 random numbers

def dot_product(a, b):

    # Check if input cases are vectors (1D arrays)
    if isinstance(a, list) and isinstance(b, list) and not all(isinstance(row, list) for row in a) and not all(
            isinstance(row, list) for row in b):
        if len(a) != len(b):
            raise ValueError("Arrays must have the same length for dot product.")
        # Calculate dot product of vectors (1D arrays)
        result = int(sum(x * y for x, y in zip(a, b)))
        return result

    # Check if input cases are matrix (2D arrays)
    elif isinstance(a, list) and isinstance(b, list) and all(isinstance(row, list) for row in a) and all(
            isinstance(row, list) for row in b) and not any(isinstance(sub, list) for row in a for sub in row) and not any(isinstance(sub, list) for row in b for sub in row):
        if len(a[0]) != len(b):
            raise ValueError("Matrix 1's columns must be equal to Matrix 2's rows for dot product.")
        result = []
        for i in range(len(a)):              # Iterate over rows of matrix 1 (a)
            row = []
            for j in range(len(b[0])):       # Iterate over columns of matrix 2 (b)
                element = 0
                for k in range(len(b)):
                    element += a[i][k] * b[k][j]
                row.append(element)
            result.append(row)
        return result

    # Check if input cases are matrix * vector (2D array * 1D array)
    elif isinstance(a, list) and isinstance(b, list) and all(isinstance(row, list) for row in a) and not all(
            isinstance(row, list) for row in b):
        if len(a[0]) != len(b):
            raise ValueError("Matrix's columns must be equal to vector's length for dot product.")
        result = []
        for i in range(len(a)):                  # Iterate over columns of matrix (a)
            element = 0
            for j in range(len(b)):               # Iterate over elements of vector (b)
                element += a[i][j] * b[j]
            result.append(element)
        return result

    # Check if input cases are vector * matrix (1D array * 2D array)
    elif isinstance(a, list) and isinstance(b, list) and not all(isinstance(row, list) for row in a) and all(
                isinstance(row, list) for row in b):
        if len(a) != len(b):
            raise ValueError("Vector's length must be equal to matrix's number of rows for dot product.")
        result = []
        for i in range(len(b[0])):            # Iterate over columns of matrix b
            element = 0
            for j in range(len(a)):           # Iterate over elements of vector a
                element += a[j] * b[j][i]
            result.append(element)
        return result

    # Check if input cases are tensors (3D arrays)
    elif (isinstance(a, list) and isinstance(b, list) and
          all(isinstance(matrix, list) for matrix in a) and
          all(isinstance(row, list) for matrix in a for row in matrix) and
          all(isinstance(matrix, list) for matrix in b) and
          all(isinstance(row, list) for matrix in b for row in matrix)):
        # Validate dimensions of the 2 tensors
        if len(a) != len(b) or any(len(a[i]) != len(b[i]) for i in range(len(a))) or any(
                len(a[i][j]) != len(b[i][j]) for i in range(len(a)) for j in range(len(a[i]))):
            raise ValueError("Tensors must have the same dimensions for dot product.")
            # Initialize the result tensor with the appropriate dimensions
        result = [[[0 for _ in range(len(b[0][0]))] for _ in range(len(a[0]))] for _ in range(len(a))]

        for i in range(len(a)):  # Iterate over the first dimension of tensor a
            for j in range(len(a[0])):  # Iterate over the second dimension (matrix rows) of tensor a
                for k in range(len(b[0][0])):  # Iterate over the third dimension (matrix columns) of tensor b
                    for l in range(len(b)):  # Iterate over the first dimension (matrices) of tensor b
                        for m in range(len(b[0])):  # Iterate over the second dimension (matrix rows) of tensor b
                            # Perform element-wise multiplication and sum for the dot product
                            result[i][j][k] += a[i][j][l] * b[l][m][k]
        return result

def inner_product(a, b):
    return dot_product(a,b)

def outer_product(a, b):
    # Check if both inputs are lists
    if not isinstance(a, list) or not isinstance(b, list):
        raise ValueError("Both inputs must be lists.")

    # Check if both inputs are 1D lists
    if any(isinstance(element, list) for element in a) or any(isinstance(element, list) for element in b):
        raise ValueError("One of the inputs is not a 1D list.")

    # Initialize matrix to store the outer product
    result = []

    # Calculate the outer product
    for i in range(len(a)):
        result.append([])
        for j in range(len(b)):
            result[i].append(a[i] * b[j])
    return result


def tensor_dot_product_along_axes(tensor1, tensor2, axis1, axis2):
    # Check if the dimensions are correct for dot product along specified axes
    if len(tensor1[0][0]) != len(tensor2):
        raise ValueError("Dimensions do not match for the specified axes.")

    # Calculate the size of the resulting tensor
    result_dims = [len(tensor1), len(tensor1[0]), len(tensor2[0][0])]
    result_dims[axis1] = len(tensor2[0])
    result_dims[axis2] = len(tensor1[axis1])

    # Initialize the resulting tensor
    result = [[[0 for _ in range(result_dims[2])] for _ in range(result_dims[1])] for _ in range(result_dims[0])]

    # Compute the dot product along the specified axes
    for i in range(len(result)):
        for j in range(len(result[0])):
            for k in range(len(result[0][0])):
                # Initialize the sum for this element
                dot_product_sum = 0
                for l in range(len(tensor1[0][0])):
                    dot_product_sum += tensor1[i][j][l] * tensor2[l][j][k]
                result[i][j][k] = dot_product_sum

    return result

def matrix_power(matrix, n):
    # Check if the input matrix is square
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("Input must be a square matrix.")
    # Initialize the result as the identity matrix of the same size
    result = [[1 if i == j else 0 for j in range(len(matrix))] for i in range(len(matrix))]
    # Early exit if the power is 0, result is identity matrix
    if n == 0:
        return result
    # Early exit if the power is 1, result is the matrix itself
    if n == 1:
        return matrix
    # Initialize a temporary variable to hold the matrix
    temp_matrix = matrix
    # Multiply the matrix by itself 'power' times
    for _ in range(n - 1):  # power - 1 because we start with matrix itself
        # Prepare a new result matrix for this iteration
        new_result = [[0 for _ in range(len(matrix))] for _ in range(len(matrix))]
        # Perform matrix multiplication
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                for k in range(len(matrix)):
                    new_result[i][j] += result[i][k] * temp_matrix[k][j]
        # Update the result matrix with the new result
        result = new_result
    return result

def kronecker_product(a, b):
    # Determine the size of the new matrix
    a_rows, a_cols = len(a), len(a[0])
    b_rows, b_cols = len(b), len(b[0])
    result_rows, result_cols = a_rows * b_rows, a_cols * b_cols
    # Create the result matrix filled with zeros
    result = [[0] * result_cols for _ in range(result_rows)]
    # Compute the Kronecker product
    for a_row in range(a_rows):
        for a_col in range(a_cols):
            # Get current element from matrix a
            a_elem = a[a_row][a_col]
            for b_row in range(b_rows):
                for b_col in range(b_cols):
                    # Compute the corresponding indices in the result matrix
                    res_row = a_row * b_rows + b_row
                    res_col = a_col * b_cols + b_col
                    # Multiply the element from a with the element from b and assign it to the result
                    result[res_row][res_col] = a_elem * b[b_row][b_col]

    return result

def cholesky_decomposition(a):
    # Check if the input matrix is square
    n = len(a)
    if any(len(row) != n for row in a):
        raise ValueError("Input matrix must be square.")
    # Initialize the result matrix with zeros
    result = [[0.0] * n for _ in range(n)]
    # Perform the Cholesky decomposition
    for i in range(n):
        for j in range(i + 1):
            temp_sum = sum(result[i][k] * result[j][k] for k in range(j))
            if i == j:  # Diagonal elements
                # The matrix must be positive definite for this to be real
                result[i][j] = (a[i][i] - temp_sum) ** 0.5
            else:
                # Off-diagonal elements
                result[i][j] = (1/result[j][j]*(a[i][j]-temp_sum))
    return result

def norm(a):
    # Check if a is a vector or a matrix
    if isinstance(a[0], list):  # Assume it's a matrix if the first element is a list
        # Compute the norm for matrix
        sum_squares = 0
        for row in a:
            for element in row:
                sum_squares += element * element
        return sum_squares ** 0.5
    else:
        # Compute the norm for vectors
        return sum(x * x for x in a) ** 0.5

def multiplicative_inverse(a):
    # Check if the matrix is square
    n = len(a)
    if any(len(row) != n for row in a):
        raise ValueError("Matrix must be square.")
    # Initialize the augmented matrix with the identity matrix
    aug_matrix = [row[:] + [0 if i != j else 1 for j in range(n)] for i, row in enumerate(a)]
    # Perform Gaussian elimination
    for i in range(n):
        # Find the pivot (largest element in the current column)
        pivot_row = max(range(i, n), key=lambda r: abs(aug_matrix[r][i]))
        if aug_matrix[pivot_row][i] == 0:
            raise ValueError("Matrix is singular and cannot be inverted.")
        # Swap the pivot row with the current row
        aug_matrix[i], aug_matrix[pivot_row] = aug_matrix[pivot_row], aug_matrix[i]
        # Normalize the pivot row
        pivot = aug_matrix[i][i]
        aug_matrix[i] = [x / pivot for x in aug_matrix[i]]
        # Eliminate the current column elements in other rows
        for j in range(n):
            if i != j:
                ratio = aug_matrix[j][i]
                aug_matrix[j] = [aug_matrix[j][k] - ratio * aug_matrix[i][k] for k in range(2 * n)]
    # Extract the inverse matrix from the augmented matrix
    result = [row[n:] for row in aug_matrix]
    return result

# Example usage:
rnd = MyRandom()
a = rnd.random(2,2)
inv_a = multiplicative_inverse(a)
import numpy as np
inv2_a = np.linalg.inv(a)
print('a = ',a)
print('inv_a = ',inv_a)
print('inv2_a = ',inv2_a)



