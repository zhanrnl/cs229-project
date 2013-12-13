import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.figure()
fig, sp = plt.subplots(2, sharex=True)
fig.set_size_inches(7,10)
plt.subplots_adjust(top=0.85,left=0.16)

cmap = plt.get_cmap('YlGn')
def to_percent(y, pos):
  return str(int(100*y)) + "%"
for i, matchdir in enumerate(["atob", "btoa"]):
  for fvec, lab, ci in zip(["", "norhythm_", "nobar_"],
      ["Full feature vector", "No rhythm data", "No bar-by-bar differentiation"],
      [0.99, 0.75, 0.5]):
    filename = "trial_data_indices_{}{}".format(fvec, matchdir)
    with open(filename) as f:
      indices = eval(f.read())
      n, b, histline = sp[i].hist(indices, label=lab, bins=max(indices), normed=True,
          cumulative=True, histtype='step')
      for line in histline:
        line.set_color(cmap(ci-0.25))
        line.set_fill(True)
        line.set_edgecolor(cmap(0.99))
        line.set_linewidth(1)
  sp[i].yaxis.set_major_formatter(FuncFormatter(to_percent))
  sp[i].grid(True)
  sp[i].set_ylabel("Percentage with rank")

sp[0].set_title("Matching A sections to given B sections")
sp[1].set_title("Matching B sections to given A sections")

sp[0].legend(loc='lower right', prop={'size':10})
plt.xlabel("Rank of correct matching section in nearest neighbor list")

plt.xlim((0,20))
#plt.suptitle("Cumulative histograms of rank of correct\nmatching sections", fontsize=20)

plt.savefig("hists.pdf")
