// Namespace module mirroring Python's `rapidfuzz.distance.JaroWinkler`.
//
// Import it as `import * as JaroWinkler from 'rapidfuzz-js/distance/JaroWinkler'` to get
// `JaroWinkler.distance(...)`, or import the named functions directly.
export {
  jaroWinklerDistance as distance,
  jaroWinklerSimilarity as similarity,
  jaroWinklerNormalizedDistance as normalizedDistance,
  jaroWinklerNormalizedSimilarity as normalizedSimilarity,
} from '../jaroWinkler.js'
