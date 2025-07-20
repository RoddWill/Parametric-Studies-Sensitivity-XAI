```markdown
# Parametric Studies Sensitivity Analysis XAI for AI Models

This Python module provides tools for sensitivity analysis and parametric studies for AI/ML models. It allows users to analyze how changes in input features impact model predictions, improving model interpretability and aiding in explainable AI (XAI).

## Features

- Parametric Study: Evaluates how predictions change when varying a single feature while keeping others constant.
- Parametric Sensitivity: Analyzes the effect of fixing one feature at specific quantiles while varying the others.
- Supports Multiple Models: Compare different AI models using visual analysis.
- Scalability: Works with various machine learning models and datasets.
- Visualization: Generates insightful plots for better interpretation.

## Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/Rod-Will/sensitivity-analysis.git
cd sensitivity-analysis
pip install -r requirements.txt
```

## Dependencies

Ensure you have the following Python libraries installed:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

Install them using:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

### 1. Import the Module

```python
from sensitivity_analysis import SensitivityAnalysis
import pandas as pd
from sklearn.linear_model import LinearRegression
```

### 2. Prepare Data and Train a Model

```python
# Example dataset
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [5, 4, 3, 2, 1],
})
target = [10, 9, 8, 7, 6]

# Train a model
model = LinearRegression().fit(data, target)

# Initialize SensitivityAnalysis
analysis = SensitivityAnalysis(model, data, ['feature1', 'feature2'])
```

### 3. Perform a Parametric Study

```python
analysis.parametric_study(models={'Linear Regression': model})
```

### 4. Perform a Parametric Sensitivity Analysis

```python
analysis.parametric_sensitivity()
```

## Functionality

### `parametric_study(models, save_path="./Output/Parametric_Study.png")`
Generates parametric study plots, illustrating how predictions change when varying one feature while keeping others constant.

#### Parameters:
- `models`: Dictionary of models `{ "Model Name": trained_model }`
- `save_path`: (Optional) Path to save the plot. Default: `"./Output/Parametric_Study.png"`

### `parametric_sensitivity(save_path="./Output/Parametric_Sensitivity.png")`
Generates sensitivity analysis plots, showing how the model's predictions change when fixing one feature at quantiles and varying others.

#### Parameters:
- `save_path`: (Optional) Path to save the plot. Default: `"./Output/Parametric_Sensitivity.png"`

## Example Output

The following plots are generated:

- Parametric Study Plot: 
  - Compares multiple models' predictions across feature values.
  - Helps understand the relationship between features and target.

- Parametric Sensitivity Plot:
  - Shows how predictions shift when different features are fixed at quantiles.
  - Identifies the most influential features.

## Testing

Run unit tests using `pytest`:

```bash
pip install pytest
pytest
```

## Contributing

Contributions are welcome! Follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to your branch (`git push origin feature-name`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Built for Explainable AI (XAI) to improve interpretability.
- Uses `scikit-learn`, `numpy`, `matplotlib`, and `pandas` for analysis and visualization.

---

Enjoy using Sensitivity Analysis for AI/ML models! 🚀  
If you find this project useful, give it a ⭐ on GitHub!
```
