def w_sum(a: list[float], b: list[float]) -> float:
    assert len(a) == len(b)
    output = 0
    for i in range(len(a)):
        output += a[i] * b[i]
    return output