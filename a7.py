import csv
from neural import NeuralNet 
import neural_net_UCI_data as nn_uci

voter_opinion = [
    ([0.9,0.6,0.8,0.3,0.1], [1.0]),
    ([0.8,0.8,0.4,0.6,0.4], [1.0]),
    ([0.7,0.2,0.4,0.6,0.3], [1.0]),
    ([0.5,0.5,0.8,0.4,0.8], [0.0]),
    ([0.3,0.1,0.6,0.8,0.8], [0.0]),
    ([0.6,0.3,0.4,0.3,0.6], [0.0])
]

von = NeuralNet(5,10,1)


file = open("rice.txt", "r").readlines()
vals = [nn_uci.parse_line(line) for line in file]
vals = nn_uci.normalize(vals)
print(vals)

