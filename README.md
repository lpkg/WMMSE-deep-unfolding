# Matrix-inverse-free implementation of the MU-MIMO WMMSE beamforming algorithm

In this GitHub repository the user can find the code used to reproduce the plots and the results in our paper [[1]](#Pellaco_MIMO). The **WMMSE algorithm** (originally proposed in [[2]](#Shi)) is a popular algorithm to address the **NP-hard non-convex weighted sum rate (WSR) maximization** problem under a total power constraint. However, its computational complexity is relatively high and, most importantly, it relies on hard-to-parallelize operations such as matrix inversions, eigendecompositions and bisection searches. In our previous paper [[3]](#Pellaco_MISO) (see master branch and ICASSP2021 branch in this repository for the code), we considered the multi-user multiple-input single-output (MU-MISO) case and effectively replaced such complex operations, but the approach that we proposed therein [[3]](#Pellaco_MISO) cannot be extended to the multi-user multiple-input multiple-output (MU-MIMO) case. Therefore, in [[1]](#Pellaco_MIMO) we consider the more challenging and most general MU-MIMO case and **we propose the first variant of the WMMSE algorithm completely free from matrix inverses and suitable to the MU-MIMO case**. We refer to this algorithm as *matrix-inverse-free WMMSE algorithm*.
By resorting to the **gradient descent** and to the **Schulz iterative approach** we effectively replace all matrix inverses in the original WMMSE algorithm with operations that can leverage parallelized implementation and hence support real-time implementation. In the paper, we formally establish that the matrix-inverse-free WMMSE algorithm **converges to a stationary point** of the NP-hard non-convex WSR maximization problem.
To leverage **deep unfolding**, we propose a more flexible variant of the matrix-inverse-free WMMSE algorithm with trainable parameters and we unfold a finite number of iterations thereof.  We train the resulting network architecture, which we refer to as *unfolded matrix-inverse-free WMMSE* (network), in order to boost the achievable performance within the fixed computational complexity.

In the jupyter notebook *Unfolded_matrix_inverse_free_WMMSE_versus_WMMSE.ipynb* the user can find:
* The implementation in Python of the WMMSE algorithm in [[2]](#Shi)
* The implementation in Python and Tensorflow 1.x of the unfolded matrix-inverse-free WMMSE in [[1]](#Pellaco_MIMO)


### Computation environment
* `Python 3` , with the following packages: `numpy` , `tensorflow`  (version 1.x), `matplotlib` , `copy` , `time`.
* `Jupyter` 

### Reference

<a name="Pellaco_MIMO"></a> [1] L. Pellaco, M. Bengtsson, J. Jaldén, "Matrix-inverse-free implementation of the MU-MIMO WMMSE beamforming algorithm," submitted.

<a name="Shi"></a> [2] Q. Shi, M. Razaviyayn, Z. Luo and C. He, "An Iteratively Weighted MMSE Approach to Distributed Sum-Utility Maximization for a MIMO Interfering Broadcast Channel," in IEEE Transactions on Signal Processing, vol. 59, no. 9, pp. 4331-4340, Sept. 2011, doi: 10.1109/TSP.2011.2147784.

<a name="Pellaco_MISO"></a> [3] L. Pellaco, M. Bengtsson, J. Jaldén, "Matrix-inverse-free deep unfolding of the weighted MMSE beamforming algorithm," IEEE Open Journal of the Communications Society, vol. 3, pp. 65-81, 2022, doi: 10.1109/OJCOMS.2021.3139858.


