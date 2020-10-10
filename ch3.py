def simplest_neuralnet(input, weight):
    prediction = input * weight
    return prediction 

def wsum(inputs, weights):
    assert(len(inputs) == len(weights))
    output  = 0.0
    for i in range(len(inputs)):
        output += (inputs[i] * weights[i])
    return output

def wsum_neuralnet(inputs, weights):
    output  = wsum(inputs, weights)
    return output

def elementwise_mult(vec_a, vec_b):
    assert(len(vec_a) == len(vec_b))
    result = [a*b for a,b in zip(vec_a, vec_b)]
    return result

def elementwise_add(vec_a, vec_b):
    assert(len(vec_a) == len(vec_b))
    result = [a+b for a,b in zip(vec_a, vec_b)]
    return result

def vec_sum(vec):
    result = 0.0
    for a in vec:
        result += a
    return result

def vec_avg(vec):
    sum = vec_sum(vec)
    return sum / len(vec)

def vect_matrix_mult(m, v):
    result = [0]*len(m)
    for i in range(len(m)):
        result[i] = (wsum(m[i], v))
    return result

matrix = [[1,0,0,0],
          [0,1,0,0],
          [0,0,1,0],
          [0,0,0,1]]




vec = [1,2,3,4]

print(vect_matrix_mult(matrix, vec))
