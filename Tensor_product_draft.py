def dot_product_tensors(tensor1, tensor2, cols1=None):
    if not all(isinstance(row, list) for row in tensor1) or not all(isinstance(row, list) for row in tensor2):
        raise ValueError("Invalid input tensors")


    # Check if dimensions are valid for tensor multiplication
    if rows1 != rows2 or cols1 != cols2:
        raise ValueError("Invalid dimensions for tensor multiplication")

        # Perform tensor multiplication
    result = [[[0 for _ in range(len(tensor2[0][0]))] for _ in range(len(tensor1[0]))] for _ in range(len(tensor1))]
    for i in range(len(tensor1)):
        for j in range(len(tensor1[0])):
            for k in range(len(tensor2[0][0])):
                for l in range(len(tensor2)):
                    for m in range(len(tensor2[0])):
                        result[i][j][k] += tensor1[i][j][l] * tensor2[l][m][k]
    return result
    return dot_product_result

# Example usage
a = [1, 3, 4, 5, 6]
tensor2 = [[7, 8], [9, 10], [11, 12]]

l = a[1][2]
# Output the result
print("Dot product of the two tensors:", l)

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