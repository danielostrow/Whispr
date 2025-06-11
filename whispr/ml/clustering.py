from typing import List

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def cluster_speakers(mfcc: np.ndarray, segments: List[tuple]) -> List[int]:
    """Cluster speech segments by averaging MFCCs, returning speaker IDs.

    Parameters
    ----------
    mfcc : np.ndarray [shape=(n_frames, n_coeffs)]
    segments : list of (start_frame, end_frame)

    Returns
    -------
    labels : list[int]
        Cluster label for each segment.
    """
    feats = []
    for start, end in segments:
        seg_vec = mfcc[start : end + 1].mean(axis=0)
        feats.append(seg_vec)
    feats = np.vstack(feats)

    # Agglomerative clustering with distance threshold to let algorithm pick #clusters
    clustering = AgglomerativeClustering(distance_threshold=8.0, n_clusters=None)
    labels = clustering.fit_predict(feats).tolist()
    return labels
