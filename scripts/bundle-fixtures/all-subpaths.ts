import * as damerau from 'rapidfuzz-js/damerau-levenshtein'
import * as fuzz from 'rapidfuzz-js/fuzz'
import * as hamming from 'rapidfuzz-js/hamming'
import * as indel from 'rapidfuzz-js/indel'
import * as jaro from 'rapidfuzz-js/jaro'
import * as jaroWinkler from 'rapidfuzz-js/jaro-winkler'
import * as lcs from 'rapidfuzz-js/lcs'
import * as levenshtein from 'rapidfuzz-js/levenshtein'
import * as osa from 'rapidfuzz-js/osa'
import * as postfix from 'rapidfuzz-js/postfix'
import * as prefix from 'rapidfuzz-js/prefix'

export const algorithms = {
  fuzz,
  levenshtein,
  indel,
  lcs,
  osa,
  damerau,
  hamming,
  jaro,
  jaroWinkler,
  prefix,
  postfix,
}
