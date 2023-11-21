# SO3 Lie group transformations and Jacobian of log map
The analysis and implementation of `SO3` `log` map, and its Jacobian (in a numerical sense). The special focus on edge cases, i.e. angles close to $0$ and $\pi$.

For details see the pdf report: [Report](SO3_transformations.pdf).

Find corresponding [Sophus](https://github.com/strasdat/Sophus) PRs [here](https://github.com/strasdat/Sophus/pull/269) and [here](https://github.com/strasdat/Sophus/pull/281)

## Contribution
* Compared 3 different formulas of log map: `log_baseline`, `log_pi`, `log_quaternion` = `log_sophus` (see jupyter notebook for details).
* Tested the above formulas of logarithmic mapping.
* Derived Jacobians of log map, i.e. `dlog(R)/dR` or `dlog(q)/dq`, in a numerical sense for each of the implementations metioned above.
* Tested Jacobians against numerical differentiation (with and without SO3 projection).
* Compared Jacobians of different implementations of log map (they turned out to be different).
* Checked the correctess of the Jacobians using chain-rule and a well-known formula of the inverse of right Jacobian, i.e. `J_r^{-1} = dlog(R+x)/dx at x=0`.
* Demonstrated the use case of the derived Jacobians in a Taylor approximation of log map.
* Demonstrated the use case of the derived Jacobians in a pose-graph optimization using ceres solver.
* Compared derived analytical Jacobians against automatically differentiated ones from jax's dual variables. 

## Conclusions
### Logarithmic mappings:
* log map in baseline implementation `log_baseline` failed for the rotation matrices corresponding to rotations by angles close to \pi (1e-8 neighborhood -> fails, 1e-7 neighborhood -> 1e-1 max abs difference, 1e-6 -> 1e-4, 1e-5 -> 1e-7)
* log map implementation adjusted for angle \pi `log_pi` gives correct results with precision up to 1e-8 max absolute difference. At angle exactly equal to \pi the log map is defined up to an overall sign.
* log map implementation using intermediate quaternion representation (matrix -> quaternion -> angle/axis) `log_quaternion` gives correct results everywhere.
* Sophus implementation of log map `log_sophus` gives correct results too. Internally it uses the "matrix to quaternion" conversion from Eigen library.

### Jacobians comparison
1. Intermediate angles:
    1. All of the Jacobians give different results.
    2. For all methods analytical Jacobian corresponds to numerical raw (except `Analytical Pi` $\neq$ `Numerical Pi raw`).
    3. `Numerical projected (norm quaternion)` $\approx$ `Analytical quaternion`, and the results are the same for all mappings (except null components of mapping around pi). 
    4. `Numerical projected (SVD)` are the same for all mappings (except null components of mapping around pi).
    
2. Small angles (Both `Quaternion` and `Baseline` methods are the best):
    1. `Pi` analytical, numerical diverge starting from angles > 1e-4. (Before 1e-4 we make case differentiation).
    2. `Quaternion` and `Baseline` methods (both analytical and numerical) do not diverge, and are changing continuously with increasing angle.
    3. `Numerical projected (norm quaternion)` prevents `Around Pi` method from diverging.
    4. `Numerical projected (SVD)` does not always prevent `Around Pi` method from diverging.
    
3. Angles close to $\pi$ (`Quaternion` method is the best):
    1. `Baseline` analytical, numerical (raw, projected) - diverge.
    2. `Pi` analytical - do not diverge. But it nullifies the components corresponding to n(i) = 0.
    3. `Pi` numerical raw - diverges.
    4. `Quaternion` analytical - does not diverge, and continuously changes with decreasing angle. 
    5. `Quaternion` numerical - diverges at close (~discretization step) proximity of $\pi$ , when the angle crosses $\pi$. 
    6. All `numerical projected` help to avoid divergence. But they also diverge at close (~discretization step) proximity of $\pi$ , when the angle crosses $\pi$.

### Correctness of Jacobians using chain rule
1. `Dx_log_x_quaternion` gives correct results everywhere (with high precision, < 1e-9 max abs diff).
2. `Dx_log_x_pi` gives inaccurate results for angles close to 0. Also it nullifies rows corresponding to exact zero components of vector $n(i)=0$.
3. `Dx_log_x_baseline` diverges at angle $\pi$, but also gives inaccurate results close to $\pi$ (already in 1e-2 proximity the inaccuracy is up to 1e-6)
4. Interestingly, the Jacobians of $\log$ map derived from different methods give different results. However, after matrix multiplication with jacobian of boxplus at zero, the results become similar or the same.
4. Numerical solutions suffer from discretization-dependent inaccuracies (up 1e-5). They also tend to diverge at angles close to $\pi$.
5. Projections reduce accuracy, but sometimes help to cope with divergence.

### Use case: Taylor approximation
1. The approximation up to precision $||x||^2$ (||x|| < 1e-1) works for all vectors, except the case when $\log(R \boxplus x)$ changes the sign of $\log(R)$

### Use case: ceres approximation
1. Checked the correctness of derived analytical Jacobians comparing them against autodifferentiated Jacobians of corresponding extended log maps.
2. Showed the issue with the baseline implementation of log map when optimizing pose-graph with large initial residuals (with angles close to \pi).

## Requirements
Libraries used:
- numpy
- Sophus (with python bindings)
- PyCeres
- jax

## References:

* Lie theory, differential calculus on manifolds:

    [[1] A micro Lie theory for state estimation in robotics. Joan Sola, Jeremie Deray, Dinesh Atchuthan, 2020](https://arxiv.org/pdf/1812.01537.pdf)

    [[2] A  Primer  on  the  Differential  Calculus  of  3D  Orientations. Michael Bloesch et al., 2016](https://arxiv.org/pdf/1606.05285.pdf)
    
    [[3] A compact formula for the derivative of a 3-D rotation in exponential coordinates. Guillermo Gallego, Anthony Yezzi, 2014](https://arxiv.org/pdf/1312.0788.pdf)
