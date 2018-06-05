Belief propagation on a Markov Random Field
with discrete pairwise potentials (MRF2).

Originally written by Kevin Murphy.
Keith Battocchi wrote the vectorized code.
Jeremy Kepner wrote the parallel MPI version.
March 2003. Last updated 7 July 2003.

There are two versions of the code: for arbitrary (general) graphs,
and for 2D lattices. The 2D lattice code is much faster.
A parallel Matlab MPI version also exists - see below.


The code assumes all potentials are pairwise.
The advantages of pairwise potentials, as opposed to general potentials, are
(1) we can compute messages using vector-matrix multiplication;
(2) we can easily specify the parameters: one potential per edge.

Note on terminology
------------------

In physics, a 2D lattice MRF is known as a Potts
model. A Potts model with binary nodes is called the Ising model.
In machine learning, an MRF2 with arbitrary structure but binary nodes
is also called a  Boltzmann machine. 



Installation
------------
1. Download the following files from 
  www.ai.mit.edu/~murphyk/Software/index.html

- BPRMF2.zip
- KPMtools.zip 
- KPMstats.zip 

2. Assuming you installed all these files in your matlab directory, in Matlab type

addpath matlab/KPMtools
addpath matlab/KPMstats
addpath matlab/BPMRF2
addpath matlab/BPMRF2/Lattice2

3. To compile the C code, type

cd matlab/BPMRF2/Lattice2
mex bp_mrf2_Chelper.c

Usage
------

There are four versions of the code:

bp_mrf2_general
bp_mrf2_lattice2
bp_mpe_mrf2_general
bp_mpe_mrf2_lattice2

1. bp_mrf2_* or bp_mpe_mrf2_*: The former computes marginal beliefs, 
the latter computes the most probable explanation (Viterbi decoding).

2. bp_*_general or bp_*_lattice2.
The former works on any graph.
(If each hidden node has a different number of possible values,
use must use cell arrays, otherwise you can use regular arrays.)
The latter works on 2D grids; all nodes must have the same number of
states.

THE LATTICE2 CODE ASSUMES POT{I,J}(XJ, XI), WHEREAS THE GENERAL
CODE ASSUMES POT{I,J}(XI, XJ). For symmetric kernels, this doesn't matter.

In addition, there are functions to compute pairwise marginals

  pairwise_bel_general
  pairwise_bel_lattice2

and an approximation to the negative log-likelihood

  bethe_mrf2_general
  bethe_mrf2_lattice


Lattice 2 implementations
-------------------------

There are several implementations of the lattice2 code from which you
can choose by specifying the 'method' argument (or calling the
functions directly).

- bp_mrf2_forloops uses for loops and is slow; it is
  designed to pedagogical/debugging purposes only.

- bp_mrf2_C is the fastest.

- bp_mrf2_vectorized is the fastest for small problems, but uses too
  much memory for big problems.

- bp_mrf2_strips is a version of vectorized that divides the lattice
  into vertical strips, and iterates over these. This has much better
  memory performance.

- bp_mrf2_local is the same as strips, but is written in a way that is
  designed to be easy to parallelize.

- bp_mrf2_mpi in the MPI directory.
  For info on Matlab MPI, see
    http://www.ll.mit.edu/MatlabMPI/files/MatlabMPI/README)


Currently (6 July 03), if each row of the kernel does not have the same sum, only
the following lattice2 implementations work correctly:  forloops,
vectorized and strips (not C or local).


Demos
-----

- demo_tree checks that BP gives exact results on a tree structured
  graph by comparing with a brute force (enumeration) method

- demo_chain checks that BP gives exact results on a 1D chain
  by comparing with a brute force (enumeration) method

- demo_lattice creates a 2D noisy image and infers the most probable
noise-less version. This demo uses medfilt2, which is in the image processing toolbox.
If you don't have this toolbox, the demo will raise an error at the
end, but you can still run it.

- test_lattice checks that the general code gives the same results as
the lattice code.


