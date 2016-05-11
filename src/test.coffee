{NeuralNet} = require './neural_net.coffee'

net = new NeuralNet 2, [4], 2, {lambda: 0.03, costDerivative: (actual, estimated) -> actual / estimated - (1 - actual) / (1 - estimated)}

getRandomInput = ->
  [Math.random() * 2 - 1, Math.random() * 2 - 1]

features = (x) ->
  return [x[0] ** 2, x[1] ** 2]

realFunction = (input) ->
  if input[0] ** 2 + input[1] ** 2 < 1
    return [1, 0]
  else
    return [0, 1]

# Training
for [0...10000]
  input = getRandomInput()
  output = realFunction(input)
  net.train features(input), output

# Testing
for [0..100]
  input = getRandomInput()
  output = realFunction input
  console.log net.estimate(features(input)), output
