import iread
import nn

import math

# Iris time!
iris_data = iread.read_data('iris/iris.data')
data = nn.TrainingDataset()
data.insize = 4
data.outsize = 1
data.l = 138
data.inputs = [x[:-1] for x in iris_data]
data.outputs = [[x[-1]] for x in iris_data]

net = nn.NeuralNetwork(1, 4, 1)

print('#################################################\n' * 3)
# Iterate until the loss isn't improving considerably
lowest_loss = 1_000_000_000
i = 0
try:
    while True:
        if i % 250 == 0:
            loss = net.obtain_data_loss(data)
            print(f'Iteration #{i} finished. Current loss: {loss}')
        i += 1
        net.learn(data, 0.001)
except KeyboardInterrupt:
    pass

print(f'Algorithm halting after {i} iterations.')

print('Done training. Will test now ...')

# 20,000 iterations seems to do the job pretty well, except that inputs that
# should be 0 are instead classed as -1. That's alright, though.
# I've seen some inaccuracies happen with 10,000 iterations, on the order of
# ten percent of the inputs.

test_data = iread.read_test_data('iris/iris.data')
for datum in test_data:
    ins = datum[:-1]
    out = net.apply(ins)[0]
    print('--------')
    print(f'Data input: {ins}')
    print(f'Expected output: {datum[-1]}')
    print(f'Actual output: {out}')
