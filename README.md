# bifit

`bifit` is a Python package designed for parameter estimation and *a posteriori* sensitivity analysis using measurements of applied controls at bifurcation points.

## Features

- **Parameter Estimation**: Estimate model parameters from measurements of external controls.
- **Sensitivity Analysis**: Perform *a posteriori* sensitivity analysis on the estimated parameters.
- **Bifurcation Analysis**: Compute bifurcation diagrams using pseudo-arclength continuation method and deflated continuation method.
- **Extensibility**: Modular design to allow easy integration of custom continuation and optimization techniques.

## Installation

To install `bifit`, clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/bifit.git
cd bifit
pip install -r requirements.txt
```

## Usage

### Example Workflow

1. **Preprocess Data**: Use the `DataPreprocessor` class to load and preprocess measurement data.
2. **Define Models**: Extend the `BaseModel` class to define your custom model.
3. **Optimization**: Use the optimization modules to estimate parameters.
4. **Analyze Results**: Perform sensitivity analysis and interpret the results.

### Code Example

**TODO**: Update this section with a correct code example.

## Documentation

Comprehensive documentation is available at [https://h-varma.github.io/bifit/](https://h-varma.github.io/bifit/).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

Special thanks to the Scientific Software Center at Heidelberg University for their courses and templates, which helped in the creation of this project.