# Deep unfolding of the weighted MMSE beamforming algorithm

This GitHub repository complements our paper [[1]](#ourpaper). The user can find the code used to reproduce the plots in the paper.
In our paper we propose the novel application of **deep unfolding** to the weighted minimum mean squre error (WMMSE) in [[2]](#WMMSE_Shi).
The WMMSE algorithm is an iterative algorithm that converges to a local solution of the weigthed sum rate maximization problem subject to a power constraint, which is known to be NP-hard. As noted in, the formulation of the WMMSE algorithm, as described in, is not suitable to be unfolded due to the matrix inversion, the eigenvale decomposition, and bisection search performed at each itearation of theh algorithm. Therefore, in our paper [[1]](#ourpaper), we propose an alternative formulation that avoids these operations. Specifically, we replace the method of Lagrangian multipliers with the **projected gradient descent (PGD) approach**. 

## Problem formulation
We consider a multiple-input single-output (MISO) interference downlink channel. The base station has M transmit antennas and sends independent data symbols to N single-antenna users. ![x_i \sim \mathcal{CN}(0,\,1)](https://render.githubusercontent.com/render/math?math=x_i%20%5Csim%20%5Cmathcal%7BCN%7D(0%2C%5C%2C1)) is the transmitted data symbol and ![\boldsymbol{h}_i \sim \mathcal{CN}(\boldsymbol{0},\,\boldsymbol{I}_M)](https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7Bh%7D_i%20%5Csim%20%5Cmathcal%7BCN%7D(%5Cboldsymbol%7B0%7D%2C%5C%2C%5Cboldsymbol%7BI%7D_M)) is the channel between user i and the base station be the channel.

## Proposed unfolded WMMSE algorithm
Algorithm 1 reports the pseudocode of the unfolded WMMSE and Fig 1 depicts the overall neural network architecture.

![](pseudocode.png)

![](unfolded_network.png)


## Computation Environment
In order to run the code in this repository the following softwares are needed:
* `Python 3` ( for reference we use Python 3.6.8 ), with the following packages:`numpy`, `tensorflow` (version 1.x - for reference we use version 1.13.1) , `matplotlib`,`copy`,`time`.
* `Jupyter` ( for reference we use version 6.0.3 )


## Reference

<a id='ourpaper'></a> [1] L. Pellaco, M. Bengtsson, J. Jaldén,"Deep unfolding of the weighted MMSE algorithm," submitted to IEEE Transactions of Signal Processing.

<a id='WMMSE_Shi'></a> [2] Q. Shi, M. Razaviyayn, Z. Luo and C. He, "An Iteratively Weighted MMSE Approach to Distributed Sum-Utility Maximization for a MIMO Interfering Broadcast Channel," in IEEE Transactions on Signal Processing, vol. 59, no. 9, pp. 4331-4340, Sept. 2011, doi: 10.1109/TSP.2011.2147784.



