// Namespace module mirroring Python's `rapidfuzz.distance.Prefix`.
//
// Import it as `import * as Prefix from 'rapidfuzz-js/distance/Prefix'` to get
// `Prefix.distance(...)`, or import the named functions directly.
export {
  prefixDistance as distance,
  prefixSimilarity as similarity,
  prefixNormalizedDistance as normalizedDistance,
  prefixNormalizedSimilarity as normalizedSimilarity,
} from '../prefix.js'
