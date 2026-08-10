// Namespace module mirroring Python's `rapidfuzz.distance.Postfix`.
//
// Import it as `import * as Postfix from 'rapidfuzz-js/distance/Postfix'` to get
// `Postfix.distance(...)`, or import the named functions directly.
export {
  postfixDistance as distance,
  postfixSimilarity as similarity,
  postfixNormalizedDistance as normalizedDistance,
  postfixNormalizedSimilarity as normalizedSimilarity,
} from '../postfix.js'
