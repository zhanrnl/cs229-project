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

if __name__ == '__main__':
  tunes = load_csv()
  tunes = parse_all_csv(tunes)
  i = 0
  while i <= len(tunes)/4000:
    tune_slice = tunes[i*4000:(i+1)*4000]
    with open('thesession-data/cpickled_parsed_{}'.format(i/4000), 'wb') as f:
      cPickle.dump(tune_slice, f)
    i += 1
