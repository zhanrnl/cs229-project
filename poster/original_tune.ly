\language "english"
#(set-global-staff-size 24)
date = #(strftime "%B %e, %Y" (localtime (current-time)))

\book {
  \header {
    title = "Cooley's (reel)"
    tagline = \date
  }


  \score {

    \new Staff \relative ef' {
      \time 4/4
      \clef treble
      \key e \minor
      \partial 4 {d4^\markup{\bold{\italic A SECTION}}}
      e8 b' b a b4 e,8 b'
      b4 a8 b d b a g
      fs d a' d, b' d, a' d,
      fs d a' d, d' a fs d
      e b' b a b4 e,8 b'
      b4 a8 b d e fs g
      a fs e cs d[ b a fs]
      d[ e fs d] e4 \bar":|.:" \break g'8^\markup{\bold{\italic B SECTION}} fs
      e b b4 e8 fs g e
      e b b4 g'8 e d b
      a4 fs8 a d, a' fs a
      a4 fs8 a d e fs g
      e b b4 e8 b g' b,
      e b b4 d8 e fs g
      a fs e cs d b a fs
      d e fs d e4 \bar ":|"
    }
  %\header {
    %piece = "An example of a tune from TheSession."
  %}
  \layout { \context { \Score \override SpacingSpanner
  #'common-shortest-duration = #(ly:make-moment 1 6) } }
}

  \paper {
    indent = 10\mm
    %ragged-last = ##t
    bookTitleMarkup = \markup {
      \override #'(baseline-skip . 4)
      \column {
        \huge \larger \bold
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
