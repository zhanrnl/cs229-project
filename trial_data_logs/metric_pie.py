from collections import OrderedDict
import matplotlib.pyplot as plt

filename = "trial_data_indices_atob"
with open(filename) as f:
  indices = eval(f.read())

d = OrderedDict([(i, 0) for i in xrange(1,11)] +
    [(s, 0) for s in ["11-15", "16-20", "21-25", "26-30", "31-50", "51-100", "other"]])
for x in indices:
  if x < 10:
    d[x+1] += 1
  elif x < 15:
    d["11-15"] += 1
  elif x < 20:
    d["16-20"] += 1
  elif x < 25:
    d["21-25"] += 1
  elif x < 30:
    d["26-30"] += 1
  elif x < 50:
    d["31-50"] += 1
  elif x < 100:
    d["51-100"] += 1
  else:
    d["other"] += 1

plt.figure(figsize=(8,8))
plt.axes([0.125, 0.06, 0.75, 0.75])
cmap = plt.get_cmap('YlGn')
patches, texts, autotexts = plt.pie(list(reversed(d.values())),
    labels=list(reversed(d.keys())), autopct='%1.0f%%', startangle=90)
for i, t in enumerate(autotexts):
  t.set_size(14)
  t.set_color('#103010')
  if len(d) - i > 4:
    t.set_visible(False)
  if i == len(d) - 1:
    t.set_color('w')
    t.set_size(28)
for t in texts:
  t.set_size(16)
  t.set_color('#103010')
for i, p in enumerate(patches):
  if i == len(d) - 1:
    p.set_color(cmap(0.95))
  else:
    p.set_color(cmap((float(i) / len(d)) * 0.85))
  p.set_linewidth(0.5)
  p.set_edgecolor('#103010')
plt.text(0.52, 0.1, "matched perfectly", color='w', fontsize=16,
    horizontalalignment='center')
plt.suptitle("Rank of correct matching A section\nto given B section in nearest neighbors", fontsize=24)
plt.title("using learned metric on full data set of 8942 tunes", fontsize=16)
plt.savefig("pie.pdf")
