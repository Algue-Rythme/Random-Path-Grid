# Random-Path-Grid
Probability of a path on a Random Grid (Percolation)

You will need the following libraries libraries:

- Numpy (for computations)
- Numba (for speed up)
- Scipy (for Linear Regression)
- Tqdm (for progress bar)
- Matplotlib (for printing)

The algorithm should be quite fast.
If you fail to install Numba, you can just comment the lines with the decorators,
and replace `prange` by `range`. However, in this case, you should expect a slower runtime.

To launch the code:

```python3 main.py```

Here are the parameters in the code you can easily change:

* the probability p
* the sequences of grid sizes N
* the number of simulation per value of N (the bigger the slower, but the bigger the better the precision will be)

Using log-log linear regression I got an estimate of the exponent: path length is almost $0.93N^{1.155}$.
Here is the plot of the lengths (the cross) as function of N:

![Empirical Estimations](https://github.com/Algue-Rythme/Random-Path-Grid/blob/main/N1000curves.png)
