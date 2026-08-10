// Namespace module mirroring Python's `rapidfuzz.distance.DamerauLevenshtein`.
//
// Import it as `import * as DamerauLevenshtein from 'rapidfuzz-js/distance/DamerauLevenshtein'` to get
// `DamerauLevenshtein.distance(...)`, or import the named functions directly.
export {
  damerauLevenshteinDistance as distance,
  damerauLevenshteinSimilarity as similarity,
  damerauLevenshteinNormalizedDistance as normalizedDistance,
  damerauLevenshteinNormalizedSimilarity as normalizedSimilarity,
} from '../damerauLevenshtein.js'
