numeric = require 'numeric'
helper = require './helper.coffee'

exports.NeuralNet = class NeuralNet
  constructor: (input, layers, outputs, options = {}) ->
    # Unpack options
    # --------------
    # @lambda: learning rate
    @lambda = options.lambda ? 0.1
    # @rectifier: rectifier function
    @rectifier = options.rectifier ? new Rectifier(helper.sigmoid, helper.dsigmoid)

    rng_bias = options.rng_bias ? Math.random
    rng_edges = options.rng_edges ? Math.random

    # Populate @biases.
    # @biases has @biases[i] being added to layers[i];
    # @biases[0] is associated with the *output* of @edges[0].
    @biases = layers.concat([outputs]).map (x) -> [1..x].map -> rng_bias()

    # @edges[i] is a matrix with layers[i] rows and layers[i - 1] columns.
    # In other words, @edges[i][j] is the input combination vector for a node.
    # This way
    #   (layers[i] x layers[i - 1]) * (layers[i - 1] x 1) => (layers[i] x 1)
    @edges = []
    all = [input].concat(layers).concat [outputs]
    for el, i in  all when i > 0
      @edges.push [1..all[i]].map -> [1..all[i - 1]].map -> rng_edges() / all[i - 1]

    return

  estimate: (input) ->
    current = input
    for edge, i in @edges
      # Linear combination
      current = numeric.dot(edge, current)

      # Biases
      current = current.map (x, j) => x + @biases[i][j]

      # Rectification
      current = current.map @rectifier.apply

    return current

  train: (input, output) ->
    # Run things forward, keeping track of the unrectified values
    # unrectified has unrectified[i] as the *output* layer of edges[i].
    # rectified has rectified[i] as the *input* layer of edges[i].
    unrectified = []
    rectified = [input]

    current = input
    for edge, i in @edges
      # Linear combination
      current = numeric.dot edge, current

      # Biases
      current = current.map (x, j) => x + @biases[i][j]

      # Record unrectified values
      unrectified.push current

      # Rectify
      current = current.map @rectifier.apply

      # Record rectified values
      rectified.push current

    # Get initial derivative dC/da[out] where a[out] is the output layer
    estimated = current
    derivativeA = estimated.map (x, i) -> output[i] - x

    # Keep track of the derivatives dC/dz[i] where z[i] is the unrectified value of layer i.
    derivatives = []

    for edge, i in @edges by -1
      # Currently, derivativeA = dC/da[i]. Find dC/dz[i].
      # ----------------------
      # Note that a[i] = rectify(z[i]), so da[i]/dz[i] = drectify(z[i]).
      # We then have:
      #   dC/dz[i] = dC/da[i] * da[i]/dz[i] = dC/da[i] * drectify(z[i])
      derivativeZ = derivativeA.map (x, j) => x * @rectifier.derivative unrectified[i][j]

      # Record the given derivative dC/dz[i]
      # derivatives has derviatves[i] as with the *output* layer of edges[i]
      # (therefore has length layers[i])
      derivatives.unshift derivativeZ

      # Find the next dC/da[i].
      # ----------------------
      # Note that z[i] = edge[i] * a[i-1] + bias[i]
      # So:
      #   dz[i]/da[i-1] = edge[i]
      #   dC/da[i - 1] = dC/dz[i] * dz[i]/da[i - 1] = dC/dz[i] * edge[i].
      #
      # We want derivativeA to be a flat vector.
      # edge[i] has dimensions (layers[i - 1] x layers[i]), dC/dz[i] has dimensions (layers[i] x 1), so we want
      #   (layers[i - 1] x 1) = (layers[i - 1] x layers[i]) * (layers[i] x 1)
      #   derviativeA = edge * derviativeZ
      derivativeA = numeric.dot(derivativeZ, edge)

    # We now get the gradients dC/dw[i][j][k] and dC/dB[i][j].
    # -------------------
    # Note that z[i] = w[i] * a[i - 1] + B[i]
    # So:
    #   dz[i]/dw[i] = a[i - 1]; dC/dw[i] = dC/dz[i] * a[i - 1]
    #   dz[i]/dB[i] = 1; dC/dB[i] = dC/dz[i]
    #
    # We want dC/dw[i] to have dimensions (layers[i] x layers[i - 1]).
    edgeUpdates = derivatives.map (derivative, i) ->
      numeric.dot numeric.transpose([derivative]), [rectified[i]]
    biasUpdates = derivatives

    # Perform gradient descent update
    for edge, i in @edges
      @edges[i] = helper.matrixAdd edge, helper.scalarMultiply @lambda, edgeUpdates[i]

    @biases = helper.matrixAdd @biases, helper.scalarMultiply @lambda, biasUpdates

    return

exports.Rectifier = class Rectifier
  constructor: (@apply, @derivative) ->
    # Approximate the derivative if it is not given
    @derivative ?= (x) =>
      (@apply(x + helper.epsilon) - @apply(x - helper.epsilon)) / (2 * helper.epsilon)
