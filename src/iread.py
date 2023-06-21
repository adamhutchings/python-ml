"""Reading in the iris dataset."""

# Lines beginning with NO: are for testing the neural network once it has been
# trained, and are not used in training.

from typing import List

# We assign 0 to setosa, 1 to versicolor, and 2 to virginica.
def flower_type(name: str) -> int:
    if name == 'Iris-setosa':
        return 0
    if name == 'Iris-versicolor':
        return 1
    if name == 'Iris-virginica':
        return 2
    raise Exception(f'Unknown flower type {name}')

def parse_line(line: str) -> List[float]:
    l = line.split(',')
    return [float(l[0]), float(l[1]), float(l[2]), float(l[3]), flower_type(l[4])]

def read_data(fname: str) -> List[List[float]]:
    with open(fname, 'r') as file:
        lines = [l.rstrip('\n') for l in file.readlines()]
        return [parse_line(line) for line in lines if not line.startswith('NO:')]

def read_test_data(fname: str) -> List[List[float]]:
    with open(fname, 'r') as file:
        lines = [l.rstrip('\n') for l in file.readlines()]
        return [parse_line(line.lstrip('NO:')) for line in lines if line.startswith('NO:')]
