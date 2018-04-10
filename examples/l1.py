

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


# locations of the simplex corners
x0 = np.array([0,0,0])
x1 = np.array([0.5,0.5*math.sqrt(3),0])
x2 = np.array([1,0,0])
x3 = np.array([0.5,0.5*math.sqrt(3) - 1.0/math.sqrt(3),math.sqrt(6)/3.0])


delta0 = np.array([1,0,0])
delta1 = np.array([0,1,0])
delta2 = np.array([0,0,1])
delta3 = np.array([0,0,0])


# p is (p[0], p[1], p[2]) adding to at most 1
# with prob of fourth outcome implied
# return point in 3d space
def p_to_pt(p):
  return p[0]*x0 + p[1]*x1 + p[2]*x2 + (1.0-sum(p))*x3


def dosplit(pts_list):
  return [p[0] for p in pts_list], [p[1] for p in pts_list], [p[2] for p in pts_list]


def my_pts_plot(ax1, pts_list, col, sty="-"):
  xs, ys, zs = dosplit(pts_list)
  ax1.plot(xs, ys, zs, sty, color=col, markersize=1)

def my_ps_plot(ax1, ps, col, sty="-"):
  pts = np.array(list(map(p_to_pt, ps)))
  my_pts_plot(ax1, pts, col, sty=sty)


# plot all lines between elements of ps_1 and ps_2
def my_lines_plot(ax1, ps_1, ps_2, col, sty="-"):
  prs = [[p1,p2] for p1 in ps_1 for p2 in ps_2 if any([e1 != e2 for e1,e2 in zip(p1,p2)])]
  for pr in prs:
    pts = np.array(list(map(p_to_pt, pr)))
    xs, ys, zs = dosplit(pts)
    ax1.plot(xs, ys, zs, sty, color=col)


def plot_simplex(ax1):
  corners = [delta0,delta1,delta2,delta3]
  my_lines_plot(ax1, corners, corners, "black", "--")


# plot the outline of the level set
def plot_levelset_exterior(ax1, me, adj1, adj2, opp, col):
  mid01 = 0.5*(me+adj1)
  mid02 = 0.5*(me+adj2)
  mid03 = 0.5*(me+opp)
  mid12 = 0.5*(adj1+adj2)
  corn0 = [me, mid12, mid03]
  corn1 = [mid01, mid02, mid03]
  my_lines_plot(ax1, corn0, corn1, col, "-")


# plot the interior of the level set
NUM_ALPHAS=30
def plot_levelset_interior(ax1, me, adj1, adj2, opp, col):
  alphas = np.linspace(0.0,1.0,NUM_ALPHAS)
  ps = np.array([a0*me + a1*adj1 + a2*adj2 + (1.0-a0-a1-a2)*opp
        for a0 in alphas for a1 in alphas for a2 in alphas if a0+a1+a2 <= 1.0 and a0-(1.0-a0-a1-a2) >= abs(a1-a2)])
  # points such that (my_prob - opp_prob) >= abs(adj1_prob - adj2_prob)
#  ps = np.array(list(filter(lambda p: p[0] - (1.0-sum(p)) >= abs(p[1] - p[2]), ps)))
  my_ps_plot(ax1, ps, col, ".")




# figure 1: simplex and outline of level set for 0
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1, projection="3d")
plot_simplex(ax1)
plot_levelset_exterior(ax1, delta0, delta1, delta2, delta3, "blue")

# figure 2: simplex, outline, interior of level set for 0
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1, projection="3d")
plot_levelset_interior(ax1, delta0, delta1, delta2, delta3, "red")
plot_simplex(ax1)
plot_levelset_exterior(ax1, delta0, delta1, delta2, delta3, "blue")

# figure 3: simplex, interior of level set for 0, 1
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1, projection="3d")
plot_levelset_interior(ax1, delta0, delta1, delta2, delta3, "red")
plot_levelset_interior(ax1, delta1, delta0, delta3, delta2, "blue")
plot_simplex(ax1)

# figure 4: simplex, interior of level set for 0, 1, 2
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1, projection="3d")
plot_levelset_interior(ax1, delta0, delta1, delta2, delta3, "red")
plot_levelset_interior(ax1, delta1, delta0, delta3, delta2, "blue")
plot_levelset_interior(ax1, delta2, delta0, delta3, delta1, "cyan")
plot_levelset_interior(ax1, delta3, delta1, delta2, delta0, "magenta")
plot_simplex(ax1)


plt.show()

