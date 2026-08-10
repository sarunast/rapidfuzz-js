// Namespace module mirroring Python's `rapidfuzz.distance.LCSseq`.
//
// Import it as `import * as LCSseq from 'rapidfuzz-js/distance/LCSseq'` to get
// `LCSseq.distance(...)`, or import the named functions directly.
export {
  lcsSeqDistance as distance,
  lcsSeqSimilarity as similarity,
  lcsSeqNormalizedDistance as normalizedDistance,
  lcsSeqNormalizedSimilarity as normalizedSimilarity,
  lcsSeqEditops as editops,
  lcsSeqOpcodes as opcodes,
} from '../lcsSeq.js'
