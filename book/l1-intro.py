weight = 0.1

def neural_network(input: float, weight: float) -> float:
    prediction = input * weight
    return prediction

numbers_of_toes = [8.5, 9.5, 10, 9]
input = numbers_of_toes[0]
pred = neural_network(input, weight)
print(pred)