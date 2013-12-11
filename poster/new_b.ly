\language "english"
#(set-global-staff-size 24)
date = #(strftime "%B %e, %Y" (localtime (current-time)))

\book {
  \header {
    tagline = \date
  }

  \score {

    \new Staff \relative ef' {
      \time 4/4
      \clef treble
      \key e \minor
      \partial 4 {fs4}
      d'8 c g g' d,4 g8 b'
      g4 c,8 a' e, b' d d
      e, e' c, a' c g' a a,
      c e, e' d, b' b a d,
      g' b g c, e4 b8 e
      c,4 a''8 b, fs a e b''
      fs g e b' a, a' a, g
      c, e fs g b4 \bar ":|"
    }
    \header {
      piece = "Randomized notes used as a starting point for the optimization."
    }
    \layout { \context { \Score \override SpacingSpanner
    #'common-shortest-duration = #(ly:make-moment 1 6) } }
  }
  \score {

    \new Staff \relative ef' {
      \time 4/4
      \clef treble
      \key e \minor
      \partial 4 {d4}
      b'8 a b g' a,4 b8 b
      a'4 b,8 b b d, fs g'
      fs a, a a fs c' b a
      d e, fs g d e e b'
      b d, e b' d4 e,8 e'
      b'4 e,,8 b' a b fs e
      b'' fs a fs, a d, fs fs'
      c, e' d, d' b4
      \bar ":|"
    }
    \header {
      piece = \markup{The new \italic{B} section, converged upon in optimization.}
    }
    \layout { \context { \Score \override SpacingSpanner
    #'common-shortest-duration = #(ly:make-moment 1 6) } }
  }

  \paper {
    indent = 0\mm
    %ragged-last = ##t
    bookTitleMarkup = \markup {
      \override #'(baseline-skip . 4)
      \column {
        %\huge \larger \bold
        %\fill-line {
          %\larger \fromproperty #'header:title
        %}
        %\fill-line {
          %\large \smaller \bold
          %\larger \fromproperty #'header:subtitle
        %}
        \fill-line { \null }
      }
    }
    scoreTitleMarkup = \markup {
      \override #'(baseline-skip . 2)
      \column {
        \large \bold \fromproperty #'header:piece
        \fill-line { \null }
      }
    }
    top-margin = 12\mm
  }
}
