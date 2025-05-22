octave --persist --eval "addpath(pwd); TDOAFDOALocBiasRedExample"

octave --gui

## Explanation

This example is purely a Monte-Carlo simulation to evaluate how the bias‐reduction step improves position/velocity estimates—it does not ingest or plot any real‐world tracking data.

It fixes a true source state (uo, u_doto) and sensor geometry.
It then generates synthetic noisy TDOA/FDOA measurements around that truth.
It calls the original WMLS solver and the bias‐reduced solver on those synthetic measurements.
Finally it computes and plots the MSE of each estimator over many runs and noise levels.
In short, there is no live or recorded “tracking” in this script—only a demonstration of error/bias reduction under controlled, simulated noise.