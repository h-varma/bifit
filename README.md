# bifit: Bifurcation-Based Parameter Estimation

`bifit` is a Python package designed for parameter estimation and *a posteriori* sensitivity analysis using measurements of applied controls at bifurcation points.

## Features

- **Parameter Estimation**: Estimate model parameters from measurements of external controls.
- **Sensitivity Analysis**: Perform *a posteriori* sensitivity analysis on the estimated parameters.
- **Bifurcation Analysis**: Compute bifurcation diagrams using pseudo-arclength continuation method and deflated continuation method.
- **Extensibility**: Modular design to allow easy integration of custom continuation and optimization techniques.

## Installation

To install `bifit`, clone the repository and install the dependencies:

```bash
git clone https://github.com/h-varma/bifit.git
cd bifit
pip install -r requirements.txt
```

## Documentation

Information about the various functions and classes can be found at [https://h-varma.github.io/bifit/](https://h-varma.github.io/bifit/). For further details on the numerical methods, their implementation and usage of the software, please refer to my [dissertation](https://archiv.ub.uni-heidelberg.de/volltextserver/36853/).

## Citation

If you use this software in your research or publications, please cite my [dissertation](https://archiv.ub.uni-heidelberg.de/volltextserver/36853/).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.