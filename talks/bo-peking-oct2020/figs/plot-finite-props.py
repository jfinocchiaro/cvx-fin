#!/usr/bin/python3

# Idea: start with a convex function of a probability distribution on three outcomes,
# pick a given number of random points in the simplex, and take hyperplanes tangent
# to the function at those points.

import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# Key parameter - controls how fine the grid is, also the running time / memory
NUM_PS = 200

colors = ['red','blue','green','yellow','orange','purple', 'beige', 'darkgreen', 'cyan', 'gold', 'magenta', 'silver', 'turquoise']
linewidth = 1.5


p1s = np.linspace(0.0, 1.0, NUM_PS)
p2s = np.linspace(0.0, 1.0, NUM_PS)
all_prs = [(p1,p2) for p1 in p1s for p2 in p2s if p1+p2 <= 1.0]

# given a probability distribution (first two probabilities)
# return x,y coordinates into the simplex where the third outcome is at (0,0),
# the first is at (1,0), and the second is at (0.5, sqrt(1.5))
def pr_to_coords(pr):
  return (pr[0]*1.0 + pr[1]*0.5, pr[1]*math.sqrt(1.5))

def get_lin_approx(f, pr):
  val = f(pr)
  gradient = ( (f((pr[0] + gradient_delta, pr[1])) - val)/gradient_delta, (f((pr[0], pr[1] + gradient_delta)) - val)/gradient_delta )
  g = lambda pr2: val + np.dot(gradient, (pr2[0] - pr[0], pr2[1] - pr[1]))
  return g


# some helper geometric functions for defining a nice convex function on the simplex
# distance from pt to line segment between a and b
def dist_to_segment(pt, a, b):
  vect = (b[0] - a[0], b[1] - a[1])
  return abs(vect[1]*pt[0] - vect[0]*pt[1] + b[0]*a[1] - b[1]*a[0]) / math.sqrt(vect[0]**2 + vect[1]**2)

def dist_to_edge_of_simplex(pr):
  pt = pr_to_coords(pr)
  corners = [(0,0), (1,0), (0.5,math.sqrt(1.5))]
  corner_pairs = [(corners[0], corners[1]), (corners[0], corners[2]), (corners[1], corners[2])]
  return min([dist_to_segment(pt, a, b) for a,b in corner_pairs])

gradient_delta = 0.000000001
def gen_example(f, num_pts):
  pts = [(random.random(), random.random()) for i in range(num_pts)]
  return [get_lin_approx(f, pt) for pt in pts]

def twonormsq(pr):
  return pr[0]**2 + pr[1]**2 + (1.0-pr[0]-pr[1])**2

# each example is a list of functions, presumably linear
# plot the pointwise max of them
my_examples = [
  [lambda pr: pr[0], lambda pr: pr[1], lambda pr: 1.0-pr[0]-pr[1]],
#  gen_example(lambda pr: 2.0  +  4.0 * twonormsq(pr), 3),
#  gen_example(lambda pr: 2.0  +  4.0 * twonormsq(pr), 5),
#  gen_example(lambda pr: 2.0  +  4.0 * twonormsq(pr), 15),
#  gen_example(lambda pr: 2.0  +  4.0 * twonormsq(pr), 30),
#  gen_example(lambda pr: 2.0  +  4.0 * twonormsq(pr), 60),

#  gen_example(lambda pr: 2.0 + 40.0 * (dist_to_edge_of_simplex(pr)**2 + 0.0 * twonormsq(pr)), 3),
#  gen_example(lambda pr: 2.0 + 40.0 * (dist_to_edge_of_simplex(pr)**2 + 0.0 * twonormsq(pr)), 5),
#  gen_example(lambda pr: 2.0 + 40.0 * (dist_to_edge_of_simplex(pr)**2 + 0.0 * twonormsq(pr)), 15),
#  gen_example(lambda pr: 2.0 + 40.0 * (dist_to_edge_of_simplex(pr)**2 + 0.0 * twonormsq(pr)), 30),
]


def plot(examples, use_colors=True):
  for ex in examples:
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1, projection="3d")
  
    f = lambda pr: max([g(pr) for g in ex])
  
    # build list of polygons
    used_gs = []
    simplex_pr_pieces = []  # each element is a report; a list of probability-pairs
    simplex_pt_pieces = []  # each element is a report; a list of (x,y) coords
  
    for g in ex: 
      my_prs = [pr for pr in all_prs if f(pr) == g(pr)]
      my_pts = [pr_to_coords(pr) for pr in my_prs]
      if len(my_pts) < 3:
        continue
      hull_inds = ConvexHull(my_pts).vertices
      if len(hull_inds) < 3:
        continue
      used_gs.append(g)
      simplex_pr_pieces.append([my_prs[i] for i in hull_inds])
      simplex_pt_pieces.append([my_pts[i] for i in hull_inds])
  
    mycolors = list(colors)
    while len(mycolors) < len(simplex_pt_pieces):
      mycolors += colors
    mycolors = mycolors[:len(simplex_pt_pieces)]
    
    if use_colors:
      col = PolyCollection(simplex_pt_pieces, facecolors=mycolors, linewidths=[linewidth]*len(mycolors), edgecolors=['black' for c in mycolors])
    else:
      col = PolyCollection(simplex_pt_pieces, facecolors=['white']*len(mycolors), linewidths=[linewidth]*len(mycolors), edgecolors=['blue' for c in mycolors])
      
    ax1.add_collection3d(col, zs=0)
  
    if use_colors:
      for g_i,g in enumerate(used_gs):
        my_pts = simplex_pt_pieces[g_i]
        ax1.plot_trisurf([pt[0] for pt in my_pts], [pt[1] for pt in my_pts], [g(pr) for pr in simplex_pr_pieces[g_i]], color=mycolors[g_i], edgecolors=mycolors[g_i], shade=False)

    else:
      for g_i,g in enumerate(used_gs):
        my_pts = simplex_pt_pieces[g_i]
        # just draw an outline of the convex hull (connecting up the last point to the first)
        ax1.plot([pt[0] for pt in my_pts] + [my_pts[0][0]], [pt[1] for pt in my_pts] + [my_pts[0][1]], [g(pr) for pr in simplex_pr_pieces[g_i]] + [g(simplex_pr_pieces[g_i][0])], color="blue")
  
  plt.show()


if __name__ == "__main__":
  plot(my_examples, True)
  plot(my_examples, False)

