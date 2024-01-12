import time

import time

class MyRandom:
    def __init__(self, seed=time.time()):
        self.seed = seed * 5
        self.a = 23
        self.b = 43
        self.c = 34

    def rand(self, min_val=0, max_val=None):
        # Generate a pseudo-random number in the range [0, self.c)
        self.seed = int((self.a * self.seed + self.b) % self.c)
        # Scale and translate the number to the desired range [min_val, max_val)
        if max_val is not None:
            return min_val + self.seed % (max_val - min_val)
        return self.seed

    def random(self, *args, min_val=0, max_val=None):
        if len(args) == 0:
            raise ValueError("At least one dimension size is required")
        # Include the range in the call to generate_array
        return self.generate_array(args, min_val=min_val, max_val=max_val)

    def generate_array(self, dimensions, depth=0, min_val=0, max_val=None):
        if depth == len(dimensions):
            return self.rand(min_val=min_val, max_val=max_val)
        # Include the range in the recursive call to generate_array
        return [self.generate_array(dimensions, depth + 1, min_val=min_val, max_val=max_val) for _ in range(dimensions[depth])]

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

def element_wise_addition(matrix1, matrix2):
    # Check if both matrices have the same dimensions
    if len(matrix1) != len(matrix2) or any(len(row1) != len(row2) for row1, row2 in zip(matrix1, matrix2)):
        raise ValueError("Both matrices must have the same dimensions.")

    # Initialize the result matrix with zeros
    result = [[0 for _ in range(len(matrix1[0]))] for _ in range(len(matrix1))]

    # Perform element-wise addition
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            result[i][j] = matrix1[i][j] + matrix2[i][j]

    return result

def element_wise_subtraction(matrix1, matrix2):
    # Check if both matrices have the same dimensions
    if len(matrix1) != len(matrix2) or any(len(row1) != len(row2) for row1, row2 in zip(matrix1, matrix2)):
        raise ValueError("Both matrices must have the same dimensions.")

    # Initialize the result matrix with zeros
    result = [[0 for _ in range(len(matrix1[0]))] for _ in range(len(matrix1))]

    # Perform element-wise subtraction
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            result[i][j] = matrix1[i][j] - matrix2[i][j]

    return result

def element_wise_mutiplication(matrix1, matrix2):
    # Check if both matrices have the same dimensions
    if len(matrix1) != len(matrix2) or any(len(row1) != len(row2) for row1, row2 in zip(matrix1, matrix2)):
        raise ValueError("Both matrices must have the same dimensions.")

    # Initialize the result matrix with zeros
    result = [[0 for _ in range(len(matrix1[0]))] for _ in range(len(matrix1))]

    # Perform element-wise subtraction
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            result[i][j] = matrix1[i][j] * matrix2[i][j]

    return result

def element_wise_division(matrix1, matrix2):
    # Check if both matrices have the same dimensions
    if len(matrix1) != len(matrix2) or any(len(row1) != len(row2) for row1, row2 in zip(matrix1, matrix2)):
        raise ValueError("Both matrices must have the same dimensions.")

    # Initialize the result matrix with zeros
    result = [[0 for _ in range(len(matrix1[0]))] for _ in range(len(matrix1))]

    # Perform element-wise subtraction
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            result[i][j] = matrix1[i][j] / matrix2[i][j]

    return result

def reshape2d_to_1d(a):
    # Flatten the 2D array into a 1D array using list comprehension
    return [element for row in a for element in row]
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

def matrix_multiplication(a, b):
    # Check if both inputs are vectors or matrix (1D or 2D arrays)
    if (isinstance(a[0], (int, float)) and isinstance(b[0], (int, float))) or (isinstance(a[0], list) and isinstance(b[0], list)):
        return dot_product(a, b)
    # If one of the arrays is 3D, matrix multiplication is not defined
    else:
        raise ValueError("Matrix multiplication is not defined for tensors (3D arrays or higher).")

def tensor_dot_product_along_axes(tensor1, tensor2, axis1, axis2):
    # Check if the dimensions are correct for dot product along specified axes
    if len(tensor1[axis1]) != len(tensor2[axis2]):
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
                for l in range(len(tensor1[0][axis1])):
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

def condition_number(a):
    # Compute the norm of the matrix
    norm_matrix = norm(a)
    # Compute the inverse of the matrix
    inv_matrix = multiplicative_inverse(a)
    # Compute the norm of the inverse matrix
    norm_inv_matrix = norm(inv_matrix)
    # Compute the condition number
    result = norm_matrix * norm_inv_matrix
    return result

def mean(a):
    if not isinstance(a, list):
        raise ValueError("input must be a vector")
    return
def determinant(a):
    # Check if the matrix is square
    n = len(a)
    if any(len(row) != n for row in a):
        raise ValueError("Matrix must be square to compute its determinant.")
    # Base case: if the matrix is 1x1, return the single element
    if len(a) == 1:
        return a[0][0]
    # Base case: if the matrix is 2x2, return the determinant
    if len(a) == 2:
        return a[0][0] * a[1][1] - a[0][1] * a[1][0]
    # Recursive case: expand along the first row
    result = 0
    for col in range(len(a)):
        # Create a submatrix excluding the first row and the current column
        submatrix = [row[:col] + row[col + 1:] for row in a[1:]]
        # Recursively calculate the determinant of the submatrix
        minor_det = determinant(submatrix)
        # Add or subtract the minor's determinant, based on column index
        result += a[0][col] * minor_det * (-1 if col % 2 else 1)
    return result

def calculate_eigen(matrix):
    if len(matrix) != 2 or len(matrix[0]) != 2:
        return "Error: This function only calculates 2x2 square arrays"

    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    trace = a + d
    determinant = a * d - b * c
    mean = trace / 2
    product = determinant

    # Check for complex eigenvalues
    discriminant = mean ** 2 - product
    if discriminant < 0:
        return (f"The eigenvalues are complex: λ1, λ2 = {mean} ± √{discriminant}, " +
                "and there are no eigenvectors for complex eigenvalues in this function.")

    # Eigenvalues
    sqrt_discriminant = discriminant ** 0.5
    lambda1 = mean + sqrt_discriminant
    lambda2 = mean - sqrt_discriminant

    # Check for cases where there are no eigenvectors
    if b == 0 and c == 0:
        # It's a diagonal matrix, eigenvectors are along the axes
        eigenvectors = [[1, 0], [0, 1]]
    elif b == 0:
        eigenvectors = [[1, 0], [d - lambda2, c]]
    elif c == 0:
        eigenvectors = [[a - lambda2, b], [0, 1]]
    else:
        # Regular case
        eigenvectors = [[b, lambda1 - a], [lambda1 - d, c]]
        # Normalize eigenvectors (not always necessary, but usually expected)
        for vec in eigenvectors:
            norm = (vec[0] ** 2 + vec[1] ** 2) ** 0.5
            vec[0] /= norm
            vec[1] /= norm

    return {'eigenvalues': (lambda1, lambda2), 'eigenvectors': eigenvectors}

