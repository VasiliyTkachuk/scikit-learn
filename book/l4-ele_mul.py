from services.ele_mul import ele_mul

weights = [0.3, 0.2, 0.9]

def neural_network(inout, weights):
    pred = ele_mul(input, weights)
    return pred

input = 0.65

print(neural_network(input, weights))