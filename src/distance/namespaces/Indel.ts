// Namespace module mirroring Python's `rapidfuzz.distance.Indel`.
//
// Import it as `import * as Indel from 'rapidfuzz-js/distance/Indel'` to get
// `Indel.distance(...)`, or import the named functions directly.
export {
  indelDistance as distance,
  indelSimilarity as similarity,
  indelNormalizedDistance as normalizedDistance,
  indelNormalizedSimilarity as normalizedSimilarity,
  indelEditops as editops,
  indelOpcodes as opcodes,
} from '../indel.js'
