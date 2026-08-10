// Namespace module mirroring Python's `rapidfuzz.distance.OSA`.
//
// Import it as `import * as OSA from 'rapidfuzz-js/distance/OSA'` to get
// `OSA.distance(...)`, or import the named functions directly.
export {
  osaDistance as distance,
  osaSimilarity as similarity,
  osaNormalizedDistance as normalizedDistance,
  osaNormalizedSimilarity as normalizedSimilarity,
} from '../osa.js'
