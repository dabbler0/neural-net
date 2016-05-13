numeric = require 'numeric'
{NeuralNet} = require './neural_net.coffee'

# # TEST 1: LINEAR REGRESSION
###

net = new NeuralNet 5, [5], 2, {lambda: 0.03, costDerivative: (actual, estimated) -> actual / estimated - (1 - actual) / (1 - estimated)}

getRandomInput = ->
  [Math.random() * 2 - 1, Math.random() * 2 - 1]

features = (x) ->
  return [x[0], x[1], x[0] ** 2, x[1] ** 2, x[0] * x[1]]

realFunction = (input) ->
  if input[0] + input[1] < 0
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
###

# # PRETEST 2: XOR
###
net = new NeuralNet 2, [4], 1, {lambda: 0.1}

getRandomInput = -> [0...2].map -> Math.round Math.random()
realFunction = (input) -> [(input[0] + input[1]) % 2]
###

# # TEST 2: ADDITION
LENGTH = 3
net = new NeuralNet LENGTH * 2, [LENGTH * LENGTH * 10, LENGTH * 10], LENGTH + 1, {lambda: 0.5}

getRandomInput = ->
  return [0...LENGTH * 2].map -> Math.round Math.random()

realFunction = (input) ->
  result = parseInt(input[0...LENGTH].join(''), 2) + parseInt(input[LENGTH...LENGTH * 2].join(''), 2)
  return ('00000000000000' + result.toString(2))[-LENGTH-1...].split('').map Number

# Training
for i in [0...1000]
  batch = net.createBatch()
  for [0...100]
    input = getRandomInput()
    output = realFunction(input)
    net.feed input, output, batch
  console.log i, batch.averageError
  net.update batch

# Testing
for [0..100]
  input = getRandomInput()
  output = realFunction input
  result = net.estimate(input)
  console.log result, output
