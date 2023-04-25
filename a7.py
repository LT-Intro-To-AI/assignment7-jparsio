from neural import NeuralNet 

voter_opinion = [
    ([0.9,0.6,0.8,0.3,0.1], [1.0]),
    ([0.8,0.8,0.4,0.6,0.4], [1.0]),
    ([0.7,0.2,0.4,0.6,0.3], [1.0]),
    ([0.5,0.5,0.8,0.4,0.8], [0.0]),
    ([0.3,0.1,0.6,0.8,0.8], [0.0]),
    ([0.6,0.3,0.4,0.3,0.6], [0.0])
]

von = NeuralNet(5,10,1)

von.train(voter_opinion)
print(von.test_with_expected(voter_opinion))

test_data = [
    [1.0,1.0,1.0,0.1,0.1],
    [0.5,0.2,0.1,0.7,0.7],
    [0.8,0.3,0.3,0.3,0.8],
    [0.8,0.3,0.3,0.8,0.3],
    [0.9,0.8,0.8,0.3,0.6]
]
for i in range(len(test_data)):
    print(f"case {i}: {von.evaluate(test_data[i])}")
