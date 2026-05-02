weight = [0.1, 0.2, 0.3, 0.4]

def w_sum(a: list[float], b: list[float]) -> float:
    assert len(a) == len(b)
    output = 0
    for i in range(len(a)):
        output += a[i] * b[i]
    return output

def neural_network(input: list[float], weight: list[float]) -> float:
    pred = w_sum(input, weight)
    return pred

toes = [8.5, 9.5, 10, 9]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]
input = [toes[0], wlrec[0], nfans[0]]
pred = neural_network(input, weight)
print(pred)

