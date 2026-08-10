// Namespace module mirroring Python's `rapidfuzz.distance.Jaro`.
//
// Import it as `import * as Jaro from 'rapidfuzz-js/distance/Jaro'` to get
// `Jaro.distance(...)`, or import the named functions directly.
export {
  jaroDistance as distance,
  jaroSimilarity as similarity,
  jaroNormalizedDistance as normalizedDistance,
  jaroNormalizedSimilarity as normalizedSimilarity,
} from '../jaro.js'
