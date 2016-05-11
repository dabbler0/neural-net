# Sigmoid
exports.sigmoid = (x) -> 1 / (1 + e ** -x)
exports.dsigmoid = (x) ->
  s = exports.sigmoid(x)
  return s * (1 - s)

# Rectified linear unit
exports.relu = (x) -> Math.max(x, 0)
exports.drelu = (x) -> if x > 0 then 1 else 0
