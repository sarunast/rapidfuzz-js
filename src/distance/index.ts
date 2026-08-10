// Mirrors Python's `rapidfuzz.distance` package.

export * as Indel from './namespaces/Indel.js'
export * as LCSseq from './namespaces/LCSseq.js'
export * as Levenshtein from './namespaces/Levenshtein.js'
export * as DamerauLevenshtein from './namespaces/DamerauLevenshtein.js'
export * as OSA from './namespaces/OSA.js'
export * as Hamming from './namespaces/Hamming.js'
export * as Jaro from './namespaces/Jaro.js'
export * as JaroWinkler from './namespaces/JaroWinkler.js'
export * as Prefix from './namespaces/Prefix.js'
export * as Postfix from './namespaces/Postfix.js'
export {
  Editops,
  Opcodes,
  type Editop,
  type EditopTag,
  type MatchingBlock,
  type Opcode,
  type OpcodeTag,
} from './editops.js'
