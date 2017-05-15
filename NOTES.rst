Speculation
===========

Need to confirm these points.  For some I'm more confidant than for others.

HOG/HOF
-------

A `cell` is the integration region for the histogram.

A `patch` is a collection of neighboring cells.  It's the ROI over which the patches are computed.

JAABA computes one patch per candidate point.  The histogram data from the entire patch is used as features for that point.
A typical patch might be 10x10 cells with each cell 40x40 px in size.

The reference computation follows Piotr Dollar's gradientHist from his vision toolbox.
JAABA always uses soft binning and does not use trilinear interpolation (AFAICT).

Piotr uses a scan to integrate the histograms along the unit-stride direction (a line).
Two line buffers are used for each side of the softbin.



Questions
=========

How to generate test data?

Understand the normalization at the end of gradientHist. 

