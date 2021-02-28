# Deep unfolding of the weighted MMSE beamforming algorithm

In this GitHub repository the user can find the code used to reproduce the plots and the results in our paper [[1]](#ourpaper).
We propose the novel application of **deep unfolding** to the weighted minimum mean square error (WMMSE) algorithm in [[2]](#WMMSE_Shi).
The WMMSE is an iterative algorithm that converges to a local solution of the weigthed sum rate maximization problem subject to a power constraint, which is known to be NP-hard. As noted in [[3]](#WMMSE_E2E), the formulation of the WMMSE algorithm, as described in [[2]](#WMMSE_Shi), is not amenable to deep unfolding due to a matrix inversion, an eigendecomposition, and a bisection search performed at each itearation. Therefore, in our paper [[1]](#ourpaper), we propose an alternative formulation that avoids these operations, while **retaining the same convergence property**. Specifically, we replace the method of Lagrange multipliers with the **projected gradient descent (PGD) approach**. In this way, the matrix inversion, the eigendecomposition, and the bisection search are replaced by simple vector operations. We refer to this variant of the original WMMSE algorithm as *unfoldable WMMSE algorithm* in order to stress its suitability to deep unfolding. By mapping each iteration of the unfoldable WMMSE algorithm to a network layer, we obtain a network architecture called **deep unfolded WMMSE**. Moreover, in order to boost the convergence of the PGD, we propose to **incorporate in the network structure Nesterov acceleration and a generalization thereof** (which we refer to as Super Nesterov) and treat such acceleration schemes as learnable structures.  

In the jupyter notebook *Deep_Unfolded_WMMSE_versus_WMMSE.ipynb* the user can find:

- The implementation in Python 3.6.8 of the WMMSE algorithm in [[2]](#WMMSE_Shi)
- The implementation in Python 3.6.8 of the zero forcing (ZF) solution
- The implementation in Python 3.6.8 of the regularized zero-forcing (RZF) solution
- The implementation in Python 3.6.8 and Tensorflow 1.13.1 of the deep unfolded WMMSE in [[1]](#ourpaper) 

In the jupyter notebooks *Nesterov_Accelerated_Deep_Unfolded_WMMSE.ipynb* and *Super_Nesterov_Accelerated_Deep_Unfolded_WMMSE.ipynb* the user can find the implementation in Python 3.6.8 and Tensorflow 1.13.1 of the accelerated deep unfolded WMMSE, with Nesterov and Super Nesteorv acceleration schemes, respectively.

The notebook cells should be run in sequential order. Note that the training time can vary from an hour to many hours, depending on the parameter settings, e.g., the number of iterations and the number of PGD steps, and on the user hardware. 


## Problem formulation
Let <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;x_i&space;\sim&space;\mathcal{CN}(0,\,1)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;x_i&space;\sim&space;\mathcal{CN}(0,\,1)" title="\small x_i \sim \mathcal{CN}(0,\,1)" /></a> be the transmitted data symbol to user <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;i" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;i" title="\small i" /></a> and let <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;\boldsymbol{h}_i&space;" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;\boldsymbol{h}_i&space;" title="\small \boldsymbol{h}_i" /></a> be the channel between the base station and user <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;i" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;i" title="\small i" /></a>.

With linear beamforming, the signal at the receiver of user <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;i" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;i" title="\small i" /></a> is 

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;y_{i}&space;=&space;{\boldsymbol{h}^{H}_i}\boldsymbol{v}_{i}x_{i}&space;&plus;&space;\sum_{j=1,j&space;\neq&space;i}^{N}{\boldsymbol{h}^{H}_i}{\boldsymbol{v}_{j}x_{j}}&space;&plus;&space;n_{i}," target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;y_{i}&space;=&space;{\boldsymbol{h}^{H}_i}\boldsymbol{v}_{i}x_{i}&space;&plus;&space;\sum_{j=1,j&space;\neq&space;i}^{N}{\boldsymbol{h}^{H}_i}{\boldsymbol{v}_{j}x_{j}}&space;&plus;&space;n_{i}," title="\small y_{i} = {\boldsymbol{h}^{H}_i}\boldsymbol{v}_{i}x_{i} + \sum_{j=1,j \neq i}^{N}{\boldsymbol{h}^{H}_i}{\boldsymbol{v}_{j}x_{j}} + n_{i}," /></a>


where <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;\boldsymbol{v}_i&space;\in&space;\mathbb{C}^{M}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;\boldsymbol{v}_i&space;\in&space;\mathbb{C}^{M}" title="\small \boldsymbol{v}_i \in \mathbb{C}^{M}" /></a> is the transmit beamformer for user <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;i" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;i" title="\small i" /></a> and where <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;n_i&space;\sim&space;\mathcal{CN}(0,\,\sigma^{2})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;n_i&space;\sim&space;\mathcal{CN}(0,\,\sigma^{2})" title="\small n_i \sim \mathcal{CN}(0,\,\sigma^{2})" /></a> is independent additive white Gaussian noise with power <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;\sigma^2" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;\sigma^2" title="\small \sigma^2" /></a>. The signal-to-interference-plus-noise-ratio (SINR) of user <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;i" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;i" title="\small i" /></a> is 


<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;\mathrm{SINR}_i&space;=&space;\frac{|\boldsymbol{h}^H_{i}\boldsymbol{v}_i|^2}{&space;\sum_{j&space;=&space;1,&space;j\neq&space;i&space;}^{N}{|\boldsymbol{h}^H_{i}\boldsymbol{v}_j|^2&space;}&plus;&space;\sigma^2}\cdot" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;\mathrm{SINR}_i&space;=&space;\frac{|\boldsymbol{h}^H_{i}\boldsymbol{v}_i|^2}{&space;\sum_{j&space;=&space;1,&space;j\neq&space;i&space;}^{N}{|\boldsymbol{h}^H_{i}\boldsymbol{v}_j|^2&space;}&plus;&space;\sigma^2}\cdot" title="\small \mathrm{SINR}_i = \frac{|\boldsymbol{h}^H_{i}\boldsymbol{v}_i|^2}{ \sum_{j = 1, j\neq i }^{N}{|\boldsymbol{h}^H_{i}\boldsymbol{v}_j|^2 }+ \sigma^2}\cdot" /></a>

The estimated data symbol at the receiver of user  <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;i" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;i" title="\small i" /></a> is <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;\hat{x}_i&space;=&space;u_{i}y_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;\hat{x}_i&space;=&space;u_{i}y_{i}" title="\small \hat{x}_i = u_{i}y_{i}" /></a>, where <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;u_i&space;\in&space;\mathbb{C}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;u_i&space;\in&space;\mathbb{C}" title="\small u_i \in \mathbb{C}" /></a> is the receiver gain.

We seek to maximize the weighted sum rate (WSR) subject to a total transmit power constraint, i.e., 

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\max_{\boldsymbol{V}}&space;\quad&space;&&space;\sum_{i&space;=&space;1}^{N}{\alpha_i\log_{2}{(&space;1&space;&plus;&space;\mathrm{SINR}_i)}}&space;\\&space;\textrm{s.t.}&space;\quad&space;&&space;\text{Tr}(\boldsymbol{V}\boldsymbol{V}^H)\leq&space;P,&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\begin{aligned}&space;\max_{\boldsymbol{V}}&space;\quad&space;&&space;\sum_{i&space;=&space;1}^{N}{\alpha_i\log_{2}{(&space;1&space;&plus;&space;\mathrm{SINR}_i)}}&space;\\&space;\textrm{s.t.}&space;\quad&space;&&space;\text{Tr}(\boldsymbol{V}\boldsymbol{V}^H)\leq&space;P,&space;\end{aligned}" title="\begin{aligned} \max_{\boldsymbol{V}} \quad & \sum_{i = 1}^{N}{\alpha_i\log_{2}{( 1 + \mathrm{SINR}_i)}} \\ \textrm{s.t.} \quad & \text{Tr}(\boldsymbol{V}\boldsymbol{V}^H)\leq P, \end{aligned}" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;\alpha_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;\alpha_{i}" title="\small \alpha_{i}" /></a> indicates the user priority (assumed to be known) and where <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;P" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;P" title="\small P" /></a> is the maximum transmit power at the base station. We assume to have perfect channel knowledge. This problem is known to be NP-hard.
We define <a href="https://www.codecogs.com/eqnedit.php?latex=\boldsymbol{H}~\triangleq&space;[\boldsymbol{h}_1,\boldsymbol{h}_2,\ldots,\boldsymbol{h}_N]^T" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\boldsymbol{H}~\triangleq&space;[\boldsymbol{h}_1,\boldsymbol{h}_2,\ldots,\boldsymbol{h}_N]^T" title="\boldsymbol{H}~\triangleq [\boldsymbol{h}_1,\boldsymbol{h}_2,\ldots,\boldsymbol{h}_N]^T" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=\boldsymbol{V}~\triangleq&space;[\boldsymbol{v}_1,\boldsymbol{v}_2,\ldots,\boldsymbol{v}_N]^T" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\boldsymbol{V}~\triangleq&space;[\boldsymbol{v}_1,\boldsymbol{v}_2,\ldots,\boldsymbol{v}_N]^T" title="\boldsymbol{V}~\triangleq [\boldsymbol{v}_1,\boldsymbol{v}_2,\ldots,\boldsymbol{v}_N]^T" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=\boldsymbol{u}&space;\triangleq&space;[u_1,&space;u_2,&space;\ldots,u_N]^T" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\boldsymbol{u}&space;\triangleq&space;[u_1,&space;u_2,&space;\ldots,u_N]^T" title="\boldsymbol{u} \triangleq [u_1, u_2, \ldots,u_N]^T" /></a>, and <a href="https://www.codecogs.com/eqnedit.php?latex=\boldsymbol{w}&space;\triangleq~[w_1,w_2,...,w_N]^T" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\boldsymbol{w}&space;\triangleq~[w_1,w_2,...,w_N]^T" title="\boldsymbol{w} \triangleq~[w_1,w_2,...,w_N]^T" /></a>.


## Proposed unfoldable WMMSE algorithm
Algorithm 1 reports the pseudocode of the unfoldable WMMSE algorithm, in which <a href="https://www.codecogs.com/eqnedit.php?latex=i" target="_blank"><img src="https://latex.codecogs.com/svg.latex?i" title="i" /></a> indexes the users, <a href="https://www.codecogs.com/eqnedit.php?latex=l" target="_blank"><img src="https://latex.codecogs.com/svg.latex?l" title="l" /></a> indexes the layers/iterations, and <a href="https://www.codecogs.com/eqnedit.php?latex=k" target="_blank"><img src="https://latex.codecogs.com/svg.latex?k" title="k" /></a> indexes the PGD steps.

![](pseudocode.png)

<a href="https://www.codecogs.com/eqnedit.php?latex=\nabla&space;f({{\boldsymbol{v}}}^{k}_i)&space;=-2\alpha_{i}w_{i}u_{i}\boldsymbol{h}_i&space;&plus;&space;2\boldsymbol{A}{\boldsymbol{v}}^{k}_i" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nabla&space;f({{\boldsymbol{v}}}^{k}_i)&space;=-2\alpha_{i}w_{i}u_{i}\boldsymbol{h}_i&space;&plus;&space;2\boldsymbol{A}{\boldsymbol{v}}^{k}_i" title="\small \nabla f({{\boldsymbol{v}}}^{k}_i) =-2\alpha_{i}w_{i}u_{i}\boldsymbol{h}_i + 2\boldsymbol{A}{\boldsymbol{v}}^{k}_i" /></a>


<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;\boldsymbol{A}&space;\triangleq&space;\sum_{i&space;=&space;1}^{N}{\alpha_{i}w_{i}|u_i|^2\boldsymbol{h}_i\boldsymbol{h}^H_i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;\boldsymbol{A}&space;\triangleq&space;\sum_{i&space;=&space;1}^{N}{\alpha_{i}w_{i}|u_i|^2\boldsymbol{h}_i\boldsymbol{h}^H_i}" title="\small \boldsymbol{A} \triangleq \sum_{i = 1}^{N}{\alpha_{i}w_{i}|u_i|^2\boldsymbol{h}_i\boldsymbol{h}^H_i}" /></a>


<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;\Pi_{\mathcal{C}}\{\boldsymbol{V}\}&space;=&space;\begin{cases}&space;\boldsymbol{V},&space;&&space;\text{if}\&space;\mathtt{Tr}(\boldsymbol{V}\boldsymbol{V}^H)\leq&space;P&space;\\&space;\frac{\boldsymbol{V}}{\left\lVert&space;\boldsymbol{V}&space;\right\rVert}\sqrt{P},&space;&&space;\text{otherwise.}&space;\end{cases}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;\Pi_{\mathcal{C}}\{\boldsymbol{V}\}&space;=&space;\begin{cases}&space;\boldsymbol{V},&space;&&space;\text{if}\&space;\mathtt{Tr}(\boldsymbol{V}\boldsymbol{V}^H)\leq&space;P&space;\\&space;\frac{\boldsymbol{V}}{\left\lVert&space;\boldsymbol{V}&space;\right\rVert}\sqrt{P},&space;&&space;\text{otherwise.}&space;\end{cases}" title="\small \Pi_{\mathcal{C}}\{\boldsymbol{V}\} = \begin{cases} \boldsymbol{V}, & \text{if}\ \mathtt{Tr}(\boldsymbol{V}\boldsymbol{V}^H)\leq P \\ \frac{\boldsymbol{V}}{\left\lVert \boldsymbol{V} \right\rVert}\sqrt{P}, & \text{otherwise.} \end{cases}" /></a>

## Deep unfolded WMMSE

Fig.1 depicts the overall network architecture.


![](network_architecture.png)
 *Fig.1. Network architecture of the deep unfolded WMMSE. It is given by L iterations of the unfoldable WMMSE algorithm. The subscripts <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;(\cdot)^{l,k}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;(\cdot)^{l,k}" title="\small (\cdot)^{l,k}" /></a> indicate the <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;k^{th}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;k^{th}" title="\small k^{th}" /></a> PGD step in the <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;l^{th}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;l^{th}" title="\small l^{th}" /></a> layer/iteration. Each layer consists of the update equation of <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;\boldsymbol{u}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;\boldsymbol{u}" title="\small \boldsymbol{u}" /></a>, denoted by <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;\boldsymbol{\Omega}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;\boldsymbol{\Omega}" title="\small \boldsymbol{\Omega}" /></a>, of the update equation of <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;\boldsymbol{w}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;\boldsymbol{w}" title="\small \boldsymbol{w}" /></a>, denoted by <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;\boldsymbol{\Psi}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;\boldsymbol{\Psi}" title="\small \boldsymbol{\Psi}" /></a>, and of K PGD steps, as depicted in the gray box. The step sizes of the truncated PGD sequence are the trainable parameters. <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;\nabla" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;\nabla" title="\small \nabla" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;\Pi_\mathcal{C}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;\Pi_\mathcal{C}" title="\small \Pi_\mathcal{C}" /></a> denote the gradient and the projection operations.*

## Computation environment
In order to run the code in this repository the following software packages are needed:
* `Python 3` (for reference we use Python 3.6.8), with the following packages:`numpy`, `tensorflow` (version 1.x - for reference we use version 1.13.1), `matplotlib`,`copy`,`time`.
* `Jupyter` (for reference we use version 6.0.3).


## Reference

<a id='ourpaper'></a> [1] L. Pellaco, M. Bengtsson, J. Jaldén, "Deep unfolding of the weighted MMSE algorithm," submitted to IEEE Transactions on Signal Processing, 2021.

<a id='WMMSE_Shi'></a> [2] Q. Shi, M. Razaviyayn, Z. Luo and C. He, "An Iteratively Weighted MMSE Approach to Distributed Sum-Utility Maximization for a MIMO Interfering Broadcast Channel," in IEEE Transactions on Signal Processing, vol. 59, no. 9, pp. 4331-4340, Sept. 2011, doi: 10.1109/TSP.2011.2147784.

<a id='WMMSE_E2E'></a> [3] H. Sun, X. Chen, Q. Shi, M. Hong, X. Fu and N. D. Sidiropoulos, "Learning to Optimize: Training Deep Neural Networks for Interference Management," in IEEE Transactions on Signal Processing, vol. 66, no. 20, pp. 5438-5453, 15 Oct.15, 2018, doi: 10.1109/TSP.2018.2866382.


