def dot_product_tensors(tensor1, tensor2):
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