"""DP-mean clustering for line in 3D space
"""
import math
import matplotlib.pyplot as plt
import numpy as np

from line import Line
from dp_mean import DPMean 


def visualize_lines(lines, color=None, points=None):
    for line in lines:
        x1 = line.x0 + line.norm * line.direction        
        plt.plot(np.array([line.x0[0, 0], x1[0, 0]]), np.array([line.x0[0, 1], x1[0, 1]]), color)
    

def visualize_clusters(clusters):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for c_id, cluster in enumerate(clusters):
        lines = cluster.assignments
        visualize_lines(lines, colors[c_id])
        plt.plot(cluster.center[0, 0], cluster.center[0, 1])



if __name__ == "__main__":
    # testing
    lines = []
    N = 8
    threshold = 0.5
    for i in range(N):
        theta = i * 2 * math.pi / N
        x = np.cos(theta)
        y = np.sin(theta)
        lines.append(Line(x0=np.array([[0, 0, 0]]), direction = np.array([[x, y, 0]]), norm=1., id_=i))
    
    for i in range(N):
        theta = i * 2 * math.pi / N
        x = np.cos(theta)
        y = np.sin(theta)
        lines.append(Line(x0=np.array([[2, 0, 0]]), direction = np.array([[x, y, 0]]), norm=1., id_=i+N))
    
    visualize_lines(lines, "r")
    plt.show()
    dp_mean = DPMean(threshold)
    dp_mean.fit(lines, init=lines[0])
    print(dp_mean.clusters)
    visualize_clusters(dp_mean.clusters)
    plt.show()




