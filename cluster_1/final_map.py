#! /usr/bin/env python
"""Make final map from the NN results."""
import numpy as np
import healpy as hp
from tqdm import tqdm
allmaps = []
indexmaps = []
for kk in tqdm(range(13), desc="Reading Files"):
    maps = np.loadtxt(f"final_map_{kk}")
    index = np.loadtxt(f"index_maps_{kk}")
    allmaps.append(maps)
    indexmaps.append(index)

clusters = hp.read_map("cluster_1.fits")
final_map = np.full_like(clusters, hp.UNSEEN)
for kk in tqdm(range(len(indexmaps)), desc="Cluster Loop"):
    for pp in tqdm(range(len(indexmaps[kk])), desc="Filling Map"):
        index = int(indexmaps[kk][pp])
        final_map[index] = allmaps[kk][pp]

hp.write_map("final_map.fits", final_map)
