# One-cut-conditional-gradient

Authors: Giacomo Cristinelli, José A. Iglesias, Daniel Walter

This module solves the control problem with Total Variation regularization

$$\min_{u\in BV(\Omega)} \frac{1}{2\alpha} |Ku-y_d|^2 + \text{TV}(u,\Omega)$$

where K is an operator associated with a linear PDE.

It employs the method described in the paper "Linear convergence of a one-cut conditional gradient method for total variation regularization". 
Preprint available at: 

Important libraries:

FEniCS (Dolfin) --version 2019.1.0 (https://fenicsproject.org/) 

Maxflow (http://pmneila.github.io/PyMaxflow/maxflow.html)

NetworkX --version 3.0 (https://networkx.org)

