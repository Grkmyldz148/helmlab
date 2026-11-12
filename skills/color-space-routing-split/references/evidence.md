# Evidence

This skill gives no performance figures. They will be published on one evidence page at helmlab.space, where each figure names its dataset, its size (n), and whether that data was used to fit, tune or design the model.

Evidence notes:
- The GenSpace vs OKLab comparison (gradient, near-gray and deutan rows) came from a ColorBench version that had defects. Its win counts are withdrawn and the comparison is being re-run; until then treat OKLab and GenSpace as alternatives.
- Helmlab MetricSpace was fitted to COMBVD, MacAdam 1974 and Helmlab's own Human Feedback data; He et al. 2022 was used in model selection. Results on these datasets are in-sample.
- The fit data of Helmlab `difference()` are thin below about L* 10 and absent above about L* 93. For comparisons across very different lightness levels, CIEDE2000 agrees better with the Munsell value scale and CIE L*.
- HDR and video rows (Jzazbz, ICtCp) follow standard practice that we have not benchmarked.
