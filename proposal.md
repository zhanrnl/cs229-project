# Comparing and Classifying Pop Music from Different Countries

*Team members: Matthew Staib, Lennart Jansson, Edward Dai*

## The Big Idea

We plan to use machine learning to distinguish modern pop music from
different countries. For example, is it possible to distinguish K-pop from
Canto-pop, Mando-pop, J-pop, or T-pop?

We think this is worthwhile and interesting for two primary reasons:

1. we can try to build upon a large amount of work on auto-tagging by trying to
improve one particular aspect (country of origin)
 
2. presuming we trust our software, we can get a numerical sense for how similar
or different pop music is. (for example, is K-pop quantitatively more like
  American pop music than J-pop is?)

### Some ideas we have/how we plan to do this

In terms of features, we were recommended by Jonathan Berger at CCRMA to look at
the
[ISMIR](http://www.ismir.net) (International Society for Music Information
Retrieval) conference and specifically at Doug Eck's work. As such, we have
found the following two papers:

- http://ismir2011.ismir.net/papers/PS6-13.pdf
- http://ismir2012.ismir.net/event/papers/553-ismir-2012.pdf

We'll start with some features (such as things related to the Mel-frequency
cepstrum) inspired from the papers and if those do not prove fruitful, we will
experiment further.

We think it'd be most interesting to try unsupervised learning, especially on
countries whose pop music may seem especially similar to the average
American. It would be especially interesting if, for example, pop music from 
Western Europe clustered together, and pop music from Eastern Europe was a 
separate cluster. By allowing the algorithm to formulate patterns and classes
for itself, we may see some quantitative similarities between genres that in
the past have only been studied qualitiatively. Algorithmically, *k*-means seems
like a good first start. We've also heard things about convolutional neural
nets, so maybe we'll try some of those?

If unsupervised learning is not fruitful even after we try to compare things
like Swedish and Indian music (which we presume to be very different), we might
try to help it along and add some supervision.
