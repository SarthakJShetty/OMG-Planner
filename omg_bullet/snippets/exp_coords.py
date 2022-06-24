#!/usr/bin/python
"""
Script to explore the similarities between 
log map in theseus and exponential coordinates in pytransform3d
More generally, is the matrix logarithm of a transformation matrix 
equivalent to a screw from screw theory?
"""
import numpy as np
import theseus as th
import pytransform3d.transformations as pt

SE3s = th.SE3.rand(30)
Ts = SE3s.to_matrix()
lms = SE3s.log_map().numpy()

lms_pt = np.zeros_like(lms)
for i in range(len(Ts)):
    T = Ts[i]

    # https://dfki-ric.github.io/pytransform3d/transformations.html#exponential-coordinates
    lm_pt = pt.exponential_coordinates_from_transform(T)

    # convention for log map in pytransform3d is to put rotational component first, 
    # hence we must swap the 3-vectors to match the theseus convention
    lm_pt = lm_pt[[3, 4, 5, 0, 1, 2]]

    lms_pt[i, :] = lm_pt

assert np.allclose(lms, lms_pt)