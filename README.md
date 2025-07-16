# Modified Causal Forest for Optimal Policy Analysis

This project uses the `mcf` Python library to train a Modified Causal Forest (MCF) model and determine an optimal policy for treatment allocation.

## Project Structure

- `main.py`: The main script to run the MCF and optimal policy analysis.
- `train.csv`: The training dataset.
- `pred.csv`: The prediction dataset.
- `model_results/`: The output directory for the trained model, results, and reports.

## Prerequisites

Before running the script, ensure you have the following libraries installed:

```bash
pip install pandas matplotlib mcf
```

## How to Run

1.  **Place your data files** (`train.csv` and `pred.csv`) in the same directory as `main.py`.
2.  **Run the script** from your terminal:

    ```bash
    python main.py
    ```

## What the Script Does

1.  **Loads Data**: Reads the training and prediction datasets from the CSV files.
2.  **Trains MCF Model**: Trains a Modified Causal Forest model using the parameters defined in the `main.py` script.
3.  **Makes Predictions**: Uses the trained model to make predictions on the prediction dataset.
4.  **Saves Outputs**: Saves the trained model, prediction results, and other artifacts to the `model_results/` directory.
5.  **Optimal Policy Analysis**: Uses the `OptimalPolicy` module to determine the best treatment allocation strategy based on the model's predictions.
6.  **Generates Report**: Creates a detailed report of the optimal policy analysis in the `model_results/` directory.

## Customization

You can customize the analysis by modifying the following sections in `main.py`:

- **File Paths**: Change the `train_data_path`, `pred_data_path`, and `output_folder` variables to match your file locations.
- **Variable Definitions**: Adjust the lists of confounding, treatment, and outcome variables.
- **MCF Parameters**: Modify the `mcf_params` dictionary to fine-tune the Modified Causal Forest model.
- **Optimal Policy Parameters**: Change the parameters for the optimal policy analysis, such as the policy features and method.
