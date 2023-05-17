import csv
from neural import NeuralNet 
import neural_net_UCI_data as nn_uci
from sklearn.model_selection import train_test_split

voter_opinion = [
    ([0.9,0.6,0.8,0.3,0.1], [1.0]),
    ([0.8,0.8,0.4,0.6,0.4], [1.0]),
    ([0.7,0.2,0.4,0.6,0.3], [1.0]),
    ([0.5,0.5,0.8,0.4,0.8], [0.0]),
    ([0.3,0.1,0.6,0.8,0.8], [0.0]),
    ([0.6,0.3,0.4,0.3,0.6], [0.0])
]



file = open("rice.csv", "r").readlines()
vals = [nn_uci.parse_line(line) for line in file]
vals = nn_uci.normalize(vals)
train, test = train_test_split(vals, test_size=0.01)
rice_net = NeuralNet(7, 10, 1)
rice_net.train(train)
results = rice_net.test_with_expected(test)

total = 0
results = [(result[1][0], result[2][0]) for result in results]
results.sort()
for result in results:
    expected = result[0]
    actual = result[1]
    difference = abs(expected - actual)
    total += difference
    print(expected, actual)
print("average difference: ", total/len(results))
