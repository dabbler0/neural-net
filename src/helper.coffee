# Sigmoid
exports.sigmoid = (x) -> 1 / (1 + Math.E ** -x)
exports.dsigmoid = (x) ->
  s = exports.sigmoid(x)
  return s * (1 - s)

# Rectified linear unit
exports.relu = (x) -> Math.max(x, 0)
exports.drelu = (x) -> if x > 0 then 1 else 0

# Simple matrix operations
exports.scalarMultiply = (scalar, matrix) ->
  matrix.map (x) -> x.map (y) -> y * scalar
exports.matrixAdd = (a, b) ->
  a.map (row, i) -> row.map (el, j) -> el + b[i][j]

# A good epsilon
exports.epsilon = 0.01
