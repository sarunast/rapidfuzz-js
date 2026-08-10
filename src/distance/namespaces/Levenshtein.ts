// Namespace module mirroring Python's `rapidfuzz.distance.Levenshtein`.
//
// Import it as `import * as Levenshtein from 'rapidfuzz-js/distance/Levenshtein'` to get
// `Levenshtein.distance(...)`, or import the named functions directly.
export {
  levenshteinDistance as distance,
  levenshteinSimilarity as similarity,
  levenshteinNormalizedDistance as normalizedDistance,
  levenshteinNormalizedSimilarity as normalizedSimilarity,
  levenshteinEditops as editops,
  levenshteinOpcodes as opcodes,
} from '../levenshtein.js'
