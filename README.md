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

## Data Preparation

The following steps were performed to prepare the datasets for modeling:
1. **Data Cleaning**: Removed rows with missing or invalid values to ensure data quality.

2. **Categorical Encoding**: Mapped categorical variables to numerical format.

3. **Income Grouping**: Continuous income variables were grouped into discrete ranges (income brackets) to reduce complexity and improve MCF stability. This step was implemented after encountering perfomance issues when the dataset size was increased.

4. **Float Conversion**: Converted all columns to float format to meet MCF requirements (e.g., `True` → `1.0`, `False` → `0.0`).

5. **Target Region Focus**: Filtered the dataset to include data from 6 selected regions.

6. **Feature Selection**: Selected relevant features based on their importance for treatment effect prediction.

7. **Train/Test Split**: Split the data into training and prediction (test) sets for model training and policy evaluation.

## Key Features

**Treatment variable**: `oct_asp_flag`: indicates whether the household received financial aid (1 = yes, 0 = no).

 **Outcome Variable**: `jan_inc`: average January income, used to measure the effect of treatment.

 **Confounding Variables**: 
 - `oct_head_inc`, `jul_head_inc`, `aug_head_inc`, `sep_head_inc`: Income of the head of the household for the given month (October, July, August, September). Values were grouped into income brackets.

 - `oct_avg_inc_opv`, `jul_avg_inc_opv`, `aug_avg_inc_opv`, `sep_avg_inc_opv`: Average household income for the given month. Values were grouped into income brackets.

 - `oct_fam_cnt`: Number of family members in the household.
 
 - `oct_fam_cat`: Assessed well-being level of the family, with values ranging from A to E:
    * A - Stable/well-off (average or above).
    * B - Satisfactory (below average).
    * C - Vulnerable (requires monitoring).
    * D – Crisis (requires assistance).
    * E – Emergency (requires urgent help).
 
 - `oct_com_re_own`: Family ownership of commercial real estate (1 = yes, 0 = no).
 
 - `oct_children`: Binary variable indicating whether family includes minor children or students under 23 years of age (1 = yes, 0 = no).
 
 - `oct_disab_child`: Presence of a person with disability from childhood or a child with disability (1 = yes, 0 = no).
 
 - `oct_overdue_debt`, `jul_overdue_debt`, `aug_overdue_debt`, `sep_overdue_debt`: Binary variable indicating whether family had overdue debt in the given month (1 = yes, 0 = no).
 
 - `oct_gov_complaint`: Binary variable indicating whether any family member was the subject of an electronic appeal or complaint submitted to the government in October (1 = yes, 0 = no).

- `oct_high_edu_adults`: Binary variable indicating whether any family member has higher education (1 = yes, 0 = no).

- `oct_fam_debt`: Binary variable indicating whether family had debt in October (1 = yes, 0 = no).

- `oct_payer_opv`: Binary variable indicating whether any family member contributes to mandatory pension payments (1 = yes, 0 = no).
 
- `oct_api_level`: Level of average per capita income (API) in the family ('Below Poverty Line', 'Low Income', 'Middle Income', 'High Income', etc.).

- `oct_regions`: Family's location by region:
    * 0 - Astana City
    * 2 - Almaty City
    * 10 - Atyrau Region
    * 17 - Turkistan Region
    * 19 - Zhambyl Region
    * 20 - Jetisu Region

- `oct_employment`: Family members employment status (e.g. 'No working members', 'One working member', 'Mulptiple working family members', etc.).

- `oct_real_estate`: Family ownership of real estate (e.g. 'No Housing', 'One House', 'Two Houses', 'Three to Five houses', etc.).

- `oct_fndr_smallbiz`: Variable indicating whether any family member is a founder of a small business (e.g. 'individual entrepreneur', 'founder of a limited liability partnership', etc.).

**Grouping Variables** are used for Group Average Treatment Effects (GATE) estimation in MCF. These define subgroups across which the model estimates heterogeneous effects. They are selected from the set of confounders:
- `oct_regions`
- `oct_high_edu_adults`
- `oct_children`
- `oct_disab_child`

