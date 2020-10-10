import numpy as np

np.random.seed(1)


def relu(x):
    return (x>0)*x

def reluderive(output):
    return output > 0


weights = np.array([0.5, 0.48, -0.7])
lr = 0.1
hidden_size = 4

streetlights = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1],
                         [0, 1, 1],
                         [1, 0, 1]])
labels = np.array([0, 1, 0, 1, 1, 0])



weights_0_1 = 2*np.random.random((3, hidden_size)) - 1
weights_1_2 = 2*np.random.random((hidden_size, 1)) - 1
for iterations in range(60):
    layer_2_error = 0
    for i in range(len(streetlights)):
        layer_0 = streetlights[i:i+1]
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        layer_2 = np.dot(layer_1, weights_1_2)
        print("weights_0_1:\n",weights_0_1)
        print("weights_1_2:\n",weights_1_2)
        layer_2_error += np.sum((layer_2-labels[i:i+1])**2)
        layer_2_delta = (layer_2 - labels[i:i+1])
        layer_1_delta = layer_2_delta.dot(weights_1_2.T)*reluderive(layer_1)
        weights_1_2 -= lr*layer_1.T.dot(layer_2_delta)
        weights_0_1 -= lr*layer_0.T.dot(layer_1_delta)
    
    if (iterations % 10) == 9:
        print("Error:" + str(layer_2_error))

