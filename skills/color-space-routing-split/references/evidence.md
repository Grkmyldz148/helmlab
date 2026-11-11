# Evidence

This skill gives no performance figures. They will be published on one evidence page at helmlab.space, where each figure names its dataset, its size (n), and whether that data was used to fit, tune or design the model.

Evidence notes:
- The GenSpace vs OKLab comparison (gradient, near-gray and deutan rows) came from a ColorBench version that had defects. Its win counts are withdrawn and the comparison is being re-run; until then treat OKLab and GenSpace as alternatives.
- Helmlab MetricSpace was fitted to COMBVD; MacAdam 1974 and He et al. 2022 were used in its fitting or model selection. Results on these datasets are in-sample.
- Helmlab `difference()` is outside its fit data for colors darker than about L* 10: there, use CIEDE2000.
- HDR and video rows (Jzazbz, ICtCp) follow standard practice that we have not benchmarked.
