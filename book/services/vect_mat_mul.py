from services.w_sum import w_sum

def vect_mat_mul(vect, matrix):
    assert len(vect) == len(matrix)
    output = [0, 0, 0]
    for i in range(len(vect)):
        output[i] = w_sum(vect, matrix[i])
    return output