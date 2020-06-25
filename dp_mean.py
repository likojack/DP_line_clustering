"""
perform dp mean clustering for line segment.
Clustering procedure
1. randomly initialize a position
2. calculate point to line distance
3. if distance is larger than a threshold,
    initialize a new cluster using the median point of a line
4. iterate 2-4 until converge.

"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
import pdb

from cluster import Cluster

def dist2pt(pt, v_0, dir_vec, norm):
    """
    calculate the point to line segment distance
    Argument:
    Return:
        dist:
        pt: corresponding points shape: [3,]
    """
    end_pts = np.concatenate([v_0, v_0 + dir_vec * norm], axis=0)
    dist_end_pts = np.sqrt(np.sum((pt - end_pts)**2, axis=1))
    p_a = pt - v_0
    len_project = p_a.dot(dir_vec.T)
    pt_on_line = v_0 + len_project * dir_vec
    dist = np.sqrt(np.sum((pt_on_line - pt)**2))
    if np.sum(np.prod(pt_on_line - end_pts, axis=0)) < 0:
        return dist
    else:
        end_id = np.argmin(dist_end_pts)
        return dist_end_pts[end_id]


class DPMean:
    def __init__(self, threshold):
        self.threshold = threshold
        self.clusters = []

    def fit(self, X, n_iter=50, init=None):
        # initialization
        center = init.get_mid_pt()
        cluster = Cluster(center)
        cluster.add(init)
        init.assigned_cluster = cluster
        self.clusters.append(cluster)

        should_stop = False
        i = 0
        while (i < n_iter) and not should_stop:
            X = np.random.permutation(X)
            # E step
            self.assign_clusters(X)
            self.remove_empty_clusters()
            # M step
            should_stop = self.recompute_mean()
            i += 1

        # remove redundant clusters with heuristic
        self.merge_clusters()

    def exist_empty_clusters(self):
        empty_clusters = [f for f in self.clusters if len(f.assignments) == 0]
        return True if len(empty_clusters) else False

    def assign_clusters(self, lines):
        num_clusters = len(self.clusters)
        for l_id, line in enumerate(lines):
            self.remove_empty_clusters()
            best_cluster = None
            min_dist = 1000000.
            for c_id, cluster in enumerate(self.clusters):
                assert len(cluster.assignments), "empty cluster should be removed"
                _dist = dist2pt(
                    cluster.center, line.x0, line.direction, line.norm
                )
                if _dist < min_dist and _dist < self.threshold:
                    min_dist = _dist
                    best_cluster = cluster
            
            if best_cluster is None:  # no matching is found
                if line.assigned_cluster is not None:
                    line.assigned_cluster.subtract(line)
                mid_pt = line.get_mid_pt()
                cluster = Cluster(center=mid_pt)
                cluster.add(line)
                self.clusters.append(cluster)
            else:
                if line.assigned_cluster is not None:
                    # remove from the old cluster
                    line.assigned_cluster.subtract(line)
                # add to the new cluster
                best_cluster.add(line)

    def remove_empty_clusters(self):
        out_clusters = []
        for cluster in self.clusters:
            if len(cluster.assignments) > 0:
                out_clusters.append(cluster)
        self.clusters = out_clusters

    def recompute_mean(self, converge_dist=1e-6):
        should_stop = 0
        for i in range(len(self.clusters)):
            if len(self.clusters[i].assignments) == 0:
                print("cluster {} is empty".format(i))
                continue
            mean_old = self.clusters[i].center
            mean_new = self.clusters[i].compute_center()
            if np.sqrt(np.sum((mean_new - mean_old) ** 2)) < converge_dist:
                should_stop += 1
        if should_stop == len(self.clusters):
            return True

    def merge(self, clusters):
        tmp_cluster = clusters[0]
        for i in range(1, len(clusters)):
            tmp_cluster.assignments.extend(clusters[i].assignments)
            tmp_cluster.A += clusters[i].A
            tmp_cluster.b += clusters[i].b
        tmp_cluster.compute_center()
        return tmp_cluster        

    def merge_clusters(self):
        """ merge redundant clusters with heuristic
        """
        merger = AgglomerativeClustering(n_clusters=None, distance_threshold=self.threshold/10)
        features = []
        for cluster in self.clusters:
            features.append(cluster.center)
        features = np.concatenate(features, axis=0)
        merger.fit(np.asarray(features))
        merge_results = merger.labels_
        uni_ids = np.unique(merge_results)
        out_clusters = []
        for id in uni_ids:
            self.clusters = np.asarray(self.clusters)
            merged_cluster = self.merge(self.clusters[id == merge_results])
            out_clusters.append(merged_cluster)
        self.clusters = out_clusters