// Namespace module mirroring Python's `rapidfuzz.distance.Hamming`.
//
// Import it as `import * as Hamming from 'rapidfuzz-js/distance/Hamming'` to get
// `Hamming.distance(...)`, or import the named functions directly.
export {
  hammingDistance as distance,
  hammingSimilarity as similarity,
  hammingNormalizedDistance as normalizedDistance,
  hammingNormalizedSimilarity as normalizedSimilarity,
  hammingEditops as editops,
  hammingOpcodes as opcodes,
} from '../hamming.js'
