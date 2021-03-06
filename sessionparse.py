import csv
import re
import cPickle
from multiprocessing import Pool
from fractions import *
from pyparsing import *

verbose = False

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
meter_eighth_beats = {
  'jig': 6,
  'reel': 8,
  'slip jig': 9,
  'hornpipe': 8,
  'polka': 4,
  'slide': 12,
  'waltz': 6,
  'barndance': 8,
  'strathspey': 8,
  'three-two': 12,
  'mazurka': 6
  }

class StrForRepr:
  def __repr__(self):
    return str(self)

class Note(StrForRepr):
  def __init__(self, pitch, dur):
    self.pitch = pitch
    self.dur = dur
  def __str__(self):
    dur_integral = self.dur.denominator == 1
    if self.dur == 1:
      dur_str = ''
    elif dur_integral:
      dur_str = self.dur.numerator
    else:
      dur_str = "{}/{}".format(self.dur.numerator, self.dur.denominator)
    return "{}{}".format(self.pitch, dur_str)

class Bar(StrForRepr):
  def __init__(self, notes, start_repeat, end_repeat, which_ending):
    self.notes = list(notes)
    self.start_repeat = start_repeat
    self.end_repeat = end_repeat
    self.which_ending = which_ending
  def __iter__(self):
    return iter(self.notes)
  def __str__(self):
    return "Bar[ {}{}{}{} ]".format(
        '<start repeat> ' if self.start_repeat else '',
        '<{} ending> '.format(self.which_ending) if \
          (self.which_ending is not None) else '',
        ' '.join(map(str, self.notes)),
        ' <end repeat>' if self.end_repeat else '',
        )

#-------------------------------------------------------------------------------

"""
Pitch handling is not precisely correct, especially with regards to accidentals
continuing to be in effect throughout a bar
"""
pitchLetters = map(chr, range(ord('A'), ord('G')+1))
pitchLetters.append('Z')
def pitch_action(ts):
  return [''.join(ts)]
accidentalP = oneOf('^ = _')
pitchP = (
    Optional(accidentalP) + 
    Suppress(Optional(Literal('~'))) +
    oneOf(pitchLetters + map(lambda c: c.lower(), pitchLetters)) +
    Optional(Literal(',') | Literal("'"))
    ).setParseAction(pitch_action)

intP = Word(nums)
def dur_action(ts):
  if ts[0] == '/':
    return [Fraction(1, 2 if len(ts) < 2 else int(ts[1]))]
  elif len(ts) >= 3 and ts[1] == '/':
    return [Fraction(int(ts[0]), int(ts[2]))]
  else:
    return [Fraction(int(ts[0]))]
durationP = (
    intP + Literal('/') + intP |
    Literal('/') + intP |
    intP + Literal('/') |
    Literal('/') |
    intP
    ).setParseAction(dur_action)

"""
Currently ignores ties ('-')
"""
def note_action(ts):
  p = ts[0]
  d = Fraction(1) if len(ts) < 2 else ts[1]
  return Note(p, d)
noteP = (
    Suppress(Optional(
      Literal('~') |
      Literal('.') |
      Literal('T')
      )) +
    pitchP +
    Optional(durationP) +
    Suppress(Optional(Literal('-')))
    ).setParseAction(note_action)

def triplet_action(ts):
  def tripletify(note):
    note.dur = note.dur * Fraction(2,3)
    return note
  return map(tripletify, ts)
tripletP = (
    Suppress(Literal('(3')) +
    noteP + noteP + noteP
  ).setParseAction(triplet_action)

def dotted_action(ts):
  if ts[1] == '>':
    return [Note(ts[0].pitch, Fraction(3,2)), Note(ts[2].pitch, Fraction(1,2))]
  else:
    return [Note(ts[0].pitch, Fraction(1,2)), Note(ts[2].pitch, Fraction(3,2))]
dottedP = (
    noteP + (
      Literal('>') | Literal('<')
    ) +
    noteP
    ).setParseAction(dotted_action)

def chord_action(ts):
  return ts[0]
chordP = (
    Suppress(Literal('[')) +
    OneOrMore(noteP) +
    Suppress(Literal(']'))
    ).setParseAction(chord_action)

"""
Ignore groups of grace notes and chord symbols
"""
chordSymbolP = Literal('"') + Word(alphas) + Literal('"')
gracesP = Literal('{') + OneOrMore(noteP) + Literal('}')
def slurred_action(ts):
  return ts
slurredP = (
    Suppress('(') + OneOrMore(
      tripletP |
      dottedP |
      noteP |
      chordP |
      Suppress(gracesP) |
      Suppress(chordSymbolP)
    ) + Suppress(')')
    ).setParseAction(slurred_action)
def bar_action(ts):
  start_repeat = False
  if ts[0] == '|:':
    start_repeat = True
    ts = ts[1:]
  end_repeat = ts[-1] == ':|'
  if type(ts[0]) is int:
    which_ending = ts[0]
    ts = ts[1:]
  else:
    which_ending = None
  bar = Bar(ts[:-1], start_repeat, end_repeat, which_ending)
  return bar
barP = (
    Optional(Literal('|:') | Suppress(Literal('|'))) +
    Optional(
      Suppress(Optional('[')) + 
      oneOf('1 2').setParseAction(lambda s: [int(s[0])])) +
    OneOrMore(
      tripletP |
      dottedP |
      noteP |
      chordP |
      Suppress(gracesP) |
      Suppress(chordSymbolP) |
      slurredP
      ) +
    ( 
      Literal(':||') |
      Literal(':|') |
      Literal('|:') |
      Literal('||') |
      Literal('|]') |
      Literal('|'))
    ).setParseAction(bar_action)

tuneP = OneOrMore(barP) + StringEnd()

def parse_abc(tune):
  abc = tune['abc']
  abc = abc.replace('\\', '')
  abc = abc.replace(':||:', ':| |:')
  abc = re.sub(r'![\w]+!', '', abc)
  abc = abc.replace('!', '')
  abc = re.sub(r'K:[\w ]+\s*', '', abc, 0, re.MULTILINE)
  return tuneP.parseString(abc)

def load_csv():
  csv.register_dialect('thesession', quoting=csv.QUOTE_MINIMAL,
      doublequote=False, escapechar='\\')
  with open('thesession-data/tunes.csv', 'r') as csvfile:
    all_tunes = list(csv.DictReader(csvfile, dialect='thesession'))
  return all_tunes

def tune_parse_fn((i, tune)):
  if i % 10 == 0:
    print 'At tune {:>6}...'.format(i)
  try:
    tune['parsed'] = list(parse_abc(tune))
  except ParseException as e:
    if verbose:
      print 'Parse error on tune', i
      print e
      print tune['abc']
  return tune
def parse_all_csv(tunes):
  pool = Pool(4)
  print 'Starting parsing...'
  tunes = pool.map_async(tune_parse_fn, enumerate(tunes), 5)
  pool.close()
  pool.join()
  #print successful, 'tunes parsed successfully,', failed, 'failed'
  return tunes.get()


def unfold_repeats(parsed):
  start_repeat = -1
  end_repeat = -1
  first_ending = -1
  second_ending = -1
  next_section = -1
  for i, bar in enumerate(parsed):
    if bar.start_repeat:
      start_repeat = i
      break
  if start_repeat == -1:
    start_repeat = 0
  for i, bar in enumerate(parsed):
    if bar.end_repeat:
      end_repeat = i
      break
  for i, bar in enumerate(parsed):
    if bar.which_ending == 1:
      first_ending = i
      break
  for i, bar in enumerate(parsed):
    if bar.which_ending == 2:
      second_ending = i
      break
  '''
  TODO: figure out why the hell these four lines were written at all.
  Currently I can't see what they do besides cause bugs with correct repeat
  unfolding
  '''
  #for i, bar in enumerate(parsed):
    #if bar.start_repeat and i > start_repeat:
      #start_repeat = i
      #break
  if second_ending != -1:
    if next_section == -1:
      next_section = second_ending + 1
    first_section = (parsed[0:start_repeat] +
        parsed[start_repeat:first_ending] +
        parsed[first_ending:second_ending] +
        parsed[start_repeat:first_ending] +
        parsed[second_ending:next_section]
        )
    return first_section + unfold_repeats(parsed[next_section:])
  elif end_repeat != -1:
    if next_section == -1:
      next_section = end_repeat + 1
    first_section = (parsed[0:start_repeat] +
        parsed[start_repeat:next_section] +
        parsed[start_repeat:next_section]
        )
    return first_section + unfold_repeats(parsed[next_section:])
  else:
    return parsed

def total_length(parsed):
  num_beats = 0
  for bar in parsed:
    for note in bar:
      num_beats += note.dur
  return num_beats

def ab_split(tune):
  """
  Adding some more heuristics to eliminate irrelevent data. Songs that are too
  long are unlikely to have the structure we want, restricting basically
  to 4 or 8 bar long A and B sections.
  B section can't start with the second iteration of a repeat. This rules out
  tunes which are just one big repeated section (probably incomplete tunes)
  """
  unfolded = unfold_repeats(tune['parsed'])
  length = int(total_length(unfolded))
  if length % (meter_eighth_beats[tune['type']] * 8) != 0:
    raise ValueError("tune doesn't have a multiple of 8 bars")
  length_in_bars = length / meter_eighth_beats[tune['type']]
  if length_in_bars > 32:
    raise ValueError("tune over 32 bars long")
  if (length_in_bars & (length_in_bars - 1) != 0) and length_in_bars > 0:
    raise ValueError("tune must be a power of 2 bars long")
  split = 0
  while total_length(unfolded[:split]) < length / 2:
    split += 1
  if total_length(unfolded[:split]) != length / 2:
    raise ValueError("could not split tune exactly into two halves by number of 8th notes")
  a = unfolded[:split]
  b = unfolded[split:]
  return (a, b)

"""
Unfolding repeats
"""
if __name__ == '__main__':
  tunes = load_csv()
  tunes = parse_all_csv(tunes[:10])
  tune = tunes[6]
  a, b = ab_split(tune)

"""
This main function reads the csv file, parses all the tunes using 4 threads,
and saves slices of 4000 tunes to pickle files. 
"""
#if __name__ == '__main__':
  #tunes = load_csv()
  #tunes = parse_all_csv(tunes)
  #i = 0
  #size = 4000
  #while i <= len(tunes)/size:
    #tune_slice = tunes[i*size:(i+1)*size]
    #with open('thesession-data/cpickled_parsed_{}'.format(i), 'wb') as f:
      #cPickle.dump(tune_slice, f)
    #i += 1
