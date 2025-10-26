
## simple monte carlo style iterative solver

python3 solv_call_bump.py

solv_call_bump : use Monte Carlo to approximate vanilla call with bump functions

a "bump" function is a smooth-ish symmetric gaussian-like sigmoid with limited extent : 
  We use a region of the cubic f(x)=3x^2 -2x^3, piecewise, to go smoothly from 0 to 1 to 0 

