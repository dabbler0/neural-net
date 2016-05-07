numeric = require 'numeric'

class NeuralNet
  constructor: (input, layers, outputs) ->
    @biases = layers.map (x) -> [1..x].map (y) -> 0
    @edges = []
    [input].concat(layers).concat([outputs]).reduce (a, b) ->
      edges.push [1..a].map (x) -> [1...b].map (y) -> 0
      return b

    return

  estimate: (input) ->
    current = input
    for edge, i in edges
      current = numeric.dot(current, edge)
      current = current.map (x, j) -> x + @biases[i][j]
      current = current.map (x) -> helper.sigmoid x
    return current

  train: (input, output) ->
    # Run things forward
    layers = []

    current = input
    for edge, i in edges
      layers.push current
      current = numeric.dot(current, edge).map (x) -> helper.sigmoid x

    estimated = current

    derivatives = []
    derivative = estimate.map (x, i) -> x - output[i]

    for edge, i in edges by -1
      derivatives.push derivative
      derivative = numeric.dot(edge, derivative).map (x) -> x * helper.dsigmoid layers[i]
