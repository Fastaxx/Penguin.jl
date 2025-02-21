# Penguin

[![Build Status](https://github.com/Fastaxx/Penguin.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Fastaxx/Penguin.jl/actions/workflows/CI.yml?query=branch%3Amain)

[![codecov](https://codecov.io/gh/Fastaxx/Penguin.jl/graph/badge.svg?token=YQUDHCTHI7)](https://codecov.io/gh/Fastaxx/Penguin.jl)

Cut-cell finite volume two phase flow solver for heat and mass transfer


## Introduction

Julia Implementation of Cut-Cell method for (now) :
- Scalar Elliptic Problems
- Scalar Parabolic Problems
- Scalar Advection-Diffusion-Reaction Problems
- Solid Moving Boundaries
- Monophasic or Diphasic Problems
- Darcy Flow Solver
- Non-prescribed Interface motion (1D)

Under development :
- Non-prescribed Interface motion (2D - 3D)
- Interface Tracking : VOF, LS, FT
- Fully Coupled Navier-Stokes Solver
- Streamfunction–vorticity solver
- Preconditionning