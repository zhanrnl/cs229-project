import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

plt.figure(figsize=(8,6))
#fig.set_size_inches(7,10)
#plt.subplots_adjust(top=0.85,left=0.16)

cmap = plt.get_cmap('YlGn')
def to_percent(y, pos):
  return str(int(100*y)) + "%"

num_features = [5, 10, 20, 30, 40, 50, 60]
fvec_type = ["", "norhythm_", "nobar_"]
fvec_label = ["Full feature vector", "No rhythm data", "No bar-by-bar differentiation"]
def str_to_arr(s):
  arrs = [ss.split(' ') for ss in s.splitlines()]
  arrs_nonempty = [[int(x) for x in xs] for xs in arrs]
  return np.array(arrs_nonempty[:2]), np.array(arrs_nonempty[2:])
def count_correct(arr):
  correct = arr[0,0] + arr[1,1]
  incorrect = arr[0,1] + arr[1,0]
  return correct, incorrect, correct + incorrect

for i, (fvt, lab) in enumerate(zip(fvec_type, fvec_label)):
  #sp[0].set_ylim(0.55, 1.0)
  #sp[0].set_xlim(0, 63)
  test_points_x = []
  train_points_x = []
  test_points_y = []
  train_points_y = []
  for nf in num_features:
    fname = "svm_result_{}{}".format(fvt, nf)
    with open(fname) as f:
      s = f.read()
      train, test = str_to_arr(s)
      train_correct, train_incorrect, train_total = count_correct(train)
      test_correct, test_incorrect, test_total = count_correct(test)
      #print nf, float(train_correct)/train_total
      train_points_x.append(nf)
      test_points_x.append(nf)
      train_points_y.append(float(train_correct)/train_total)
      test_points_y.append(float(test_correct)/test_total)
  lines = plt.plot(test_points_x, test_points_y, 'ro-', label=lab + " (test)")
  for line in lines:
    line.set_color(cmap(0.9 - 0.3 * i))
    line.set_linewidth(4)
  lines = plt.plot(train_points_x, train_points_y, 'ro--', label=lab + " (train)")
  for line in lines:
    line.set_color(cmap(0.9 - 0.3 * i))
    line.set_linewidth(2)

plt.axes().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.ylim(0.4, 1.0)
plt.xlim(0, 65)
plt.xlabel("Number of features in PCA")
plt.ylabel("Percentage correct")
plt.legend(loc='lower right', prop={'size':10})
plt.suptitle("A/B classification results with SVM", fontsize=20)
plt.savefig("svm.pdf")
