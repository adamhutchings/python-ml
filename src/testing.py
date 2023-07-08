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

nets = []
for i in range(10):
    nets.append(nn.NeuralNetwork(1, 4, 1))

print('Training and selecting initial models ...')
r = 500
for i in range(r):
    if i % (r / 10) == 0:
        print(f'{round(i * 100 / r)}% done.')
    for net in nets:
        net.learn(data, 0.001)

net = nn.NeuralNetwork(1, 4, 1)
for n in nets:
    if n.obtain_data_loss(data) < net.obtain_data_loss(data):
        net = n

print(f'Obtained best net out of 10 with loss of {round(net.obtain_data_loss(data), 3)}.')

# 20,000 iterations seems to do the job pretty well, except that inputs that
# should be 0 are instead classed as -1. That's alright, though.
# I've seen some inaccuracies happen with 10,000 iterations, on the order of
# ten percent of the inputs.

test_data = iread.read_test_data('iris/iris.data')
inaccuracies = 0
for datum in test_data:
    ins = datum[:-1]
    out = net.apply(ins)[0]
    # print('--------')
    #print(f'Data input: {ins}')
    #print(f'Expected output: {datum[-1]}')
    #print(f'Actual output: {out}')
    if round(datum[-1]) != round(out):
        inaccuracies += 1
print(f'In the model, {inaccuracies} out of {len(test_data)} were inaccurate.')
