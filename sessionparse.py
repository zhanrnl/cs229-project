import pickle
import csv
from music21 import *

meter_dict = {
  'jig': '6/8',
  'reel': '4/4',
  'slip jig': '9/8',
  'hornpipe': '4/4',
  'polka': '2/4',
  'slide': '12/8',
  'waltz': '3/4',
  'barndance': '4/4',
  'strathspey': '4/4',
  'three-two': '3/2',
  'mazurka': '3/4'
  }

def parse_abc(abcobj):
  """
  Given a row of the csv file, parses into a music21 stream
  """
  abcstr = "T: {} ({})\nM: {}\nL: 1/8\nK: {}\n{}".format(abcobj['name'],
      abcobj['type'], meter_dict[abcobj['type']], abcobj['mode'],
      abcobj['abc'])
  return converter.parseData(abcstr, None, 'abc')

def parse_all_csv(n=None):
  """
  Reads all rows of the csv and as an iterator yields each parsed tune music21
  stream. The parsed streams can be serialized with converter.freezeStr, but
  this creates hideously huge binary blobs (hundreds KB each), so we should
  just do processing on each tune one by one instead of trying to serialize
  parse results
  """
  csv.register_dialect('thesession', quoting=csv.QUOTE_MINIMAL,
      doublequote=False, escapechar='\\')
  with open('thesession-data/tunes.csv', 'r') as csvfile:
    all_tunes = list(csv.DictReader(csvfile, dialect='thesession'))

  for tune in (all_tunes if n is None else all_tunes[:n]):
    try:
      tune['score'] = parse_abc(tune)
      yield tune
    except:
      pass

if __name__ == '__main__':
  # silly example
  for tune in parse_all_csv(20):
    pass
    #print tune
    #tune['score'].show()
