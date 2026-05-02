from services.w_sum import w_sum

weight = [0.1, 0.2, 0]

def neural_network(input: list[float], weight: list[float]) -> float:
    pred = w_sum(input, weight)
    return pred

toes = [8.5, 9.5, 10, 9]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]
# input = [toes[0], wlrec[0], nfans[0]]
input = [toes[1], wlrec[1], nfans[1]]
pred = neural_network(input, weight)
print(pred)


def elementwise_multiplication(vec_a: list[float], vec_b: list[float]) -> list[float]:
    assert len(vec_a) == len(vec_b)
    output = []
    for i in range(len(vec_a)):
        output.append(vec_a[i] * vec_b[i])
    return output

print(elementwise_multiplication([1, 2, 3], [4, 5, 6]))

def elementwise_addition(vec_a: list[float], vec_b: list[float]) -> list[float]:
    assert len(vec_a) == len(vec_b)
    output = []
    for i in range(len(vec_a)):
        output.append(vec_a[i] + vec_b[i])
    return output

print(elementwise_addition([1, 2, 3], [4, 5, 6]))

def vector_sum(vec: list[float]) -> float:
    output = 0
    for i in range(len(vec)):
        output += vec[i]
    return output

print(vector_sum([1, 2, 3]))

def vector_average(vec: list[float]) -> float:
    output = 0
    for i in range(len(vec)):
        output += vec[i]
    return output / len(vec)

print(vector_average([1, 2, 3]))
