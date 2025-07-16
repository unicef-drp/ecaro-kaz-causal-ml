# This script trains a Modified Causal Forest (MCF) model to estimate heterogeneous
# treatment effects and then uses the results to determine an optimal policy for treatment
# allocation.

# The script performs the following steps:
# 1.  Loads training and prediction data.
# 2.  Defines the parameters for the MCF model.
# 3.  Trains the MCF model on the training data.
# 4.  Makes predictions on a separate prediction dataset.
# 5.  Saves the trained model and the prediction results.
# 6.  Loads the saved results and model.
# 7.  Uses the OptimalPolicy module to find the optimal treatment allocation.
# 8.  Generates a report of the optimal policy results.

import os
import pickle
import pandas as pd
from mcf.mcf_functions import ModifiedCausalForest
from mcf.optpolicy_functions import OptimalPolicy
from mcf.reporting import McfOptPolReport
from sklearn.model_selection import train_test_split



def save_mcf_outputs(
    tree_df: pd.DataFrame = None,
    fill_y_df: pd.DataFrame = None,
    results: dict = None,
    tag: str = "run1"
) -> None:
    """
    Saves MCF artifacts to disk.

    Args:
        tree_df (pd.DataFrame, optional): The trained forest structure. Defaults to None.
        fill_y_df (pd.DataFrame, optional): The honest 'fill-Y' sample. Defaults to None.
        results (dict, optional): The prediction/analysis dictionaries. Defaults to None.
        tag (str, optional): A string to identify the run. Defaults to "run1".
    """
    try:
        os.makedirs(tag, exist_ok=True)

        if tree_df is not None:
            tree_df.to_csv(os.path.join(tag, "trees.csv"), index=False)

        if fill_y_df is not None:
            fill_y_df.to_pickle(os.path.join(tag, "fill_y.pkl"))

        if results is not None:
            for key, value in results.items():
                if isinstance(value, pd.DataFrame):
                    value.to_csv(os.path.join(tag, f"{key}.csv"), index=False)
            with open(os.path.join(tag, "results_dict.pkl"), "wb") as f:
                pickle.dump(results, f)
    except (IOError, pickle.PicklingError) as e:
        print(f"Error saving MCF outputs: {e}")


def main():
    """
    Main function to run the MCF and Optimal Policy analysis.
    """
    # --- Configuration ---

    # File paths
    train_data_path = 'train.csv'
    pred_data_path = 'pred.csv'
    output_folder = "model_results"

    # Confounding variables
    confounder_ordered = [
        'oct_head_inc', 'oct_avg_inc_opv', 'oct_fam_cnt', 'oct_fam_cat',
        'oct_com_re_own', 'oct_children', 'oct_disab_child', 'oct_overdue_debt',
        'oct_gov_complaint', 'jul_overdue_debt', 'aug_overdue_debt',
        'sep_overdue_debt', 'jul_head_inc', 'oct_high_edu_adults',
        'oct_fam_debt', 'oct_payer_opv', 'jul_avg_inc_opv', 'aug_head_inc',
        'aug_avg_inc_opv', 'sep_head_inc', 'sep_avg_inc_opv', 'oct_api_level'
    ]
    confounder_unordered = [
        'oct_regions', 'oct_employment', 'oct_real_estate', 'oct_fndr_smallbiz'
    ]

    # Variables for MCF
    treatment_variable = ['oct_asp_flag']
    outcome_variable = 'jan_avg_inc_opv'
    policy_features_ordered = ['oct_high_edu_adults', 'oct_children', 'oct_disab_child']
    policy_features_unordered = ['oct_regions']

    # MCF parameters
    mcf_params = {
        'cs_type': 1,
        'cs_min_p': 0.01,
        'cs_max_del_train': 0.5,
        'var_x_name_ord': confounder_ordered,
        'var_x_name_unord': confounder_unordered,
        'var_y_name': outcome_variable,
        'var_d_name': treatment_variable,
        'gen_iate_eff': True,
        'cf_m_share_max': 0.4,
        'cf_m_share_min': 0.2,
        'p_iate_se': True,
        'p_ci_level': 0.9,
        'var_z_name_ord': policy_features_ordered,
        'var_z_name_unord': policy_features_unordered
    }

    # Optimal policy parameters
    opt_policy_method = 'policy tree'
    opt_policy_features_ordered = ['oct_children']
    opt_policy_features_unordered = ['oct_regions']

    # --- Data Loading ---
    try:
        df_train = pd.read_csv(train_data_path)
        df_pred = pd.read_csv(pred_data_path)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # --- MCF Model Training ---
    print("Training Modified Causal Forest...")
    mcf_model = ModifiedCausalForest(**mcf_params)
    tree_df, fill_y_df, _ = mcf_model.train(df_train)
    print("MCF training complete.")

    # --- Prediction ---
    print("Making predictions...")
    pred, eval_set = train_test_split(df_pred, test_size=0.3, random_state=42)
    results, _ = mcf_model.predict(pred)
    results_oos, _ = mcf_model.predict(eval_set)
    print("Prediction complete.")

    # --- Saving Outputs ---
    print("Saving model and results...")
    save_mcf_outputs(tree_df, fill_y_df, tag=os.path.join(output_folder, "train"))
    with open(os.path.join(output_folder, "mcf_model.pkl"), "wb") as f:
        pickle.dump(mcf_model, f)
    save_mcf_outputs(results=results, tag=os.path.join(output_folder, "results"))
    save_mcf_outputs(results=results_oos, tag=os.path.join(output_folder, "results_oos"))
    print("Outputs saved.")

    # --- Optimal Policy ---
    print("Determining optimal policy...")
    # Load saved results for optimal policy
    try:
        with open(os.path.join(output_folder, "results", "results_dict.pkl"), 'rb') as f:
            results_loaded = pickle.load(f)
        with open(os.path.join(output_folder, "results_oos", "results_dict.pkl"), 'rb') as f:
            results_oos_loaded = pickle.load(f)
        with open(os.path.join(output_folder, "mcf_model.pkl"), 'rb') as f:
            mcf_model_loaded = pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"Error loading saved model/results: {e}")
        return

    data_train_pt = results_loaded['iate_data_df']
    oos_df = results_oos_loaded['iate_data_df']

    potential_outcomes = [
        f"{outcome_variable.casefold()}_lc{i}_un_lc_pot_eff" for i in [0, 1]
    ]

    opt_policy = OptimalPolicy(
        var_d_name=treatment_variable,
        var_x_name_ord=opt_policy_features_ordered,
        var_x_name_unord=opt_policy_features_unordered,
        var_polscore_name=potential_outcomes,
        gen_method=opt_policy_method
    )

    alloc_train_df, _, _ = opt_policy.solve(data_train_pt, data_title='Training PT data')
    opt_policy.evaluate(alloc_train_df, data_train_pt, data_title='Training PT data')
    alloc_eva_df, _ = opt_policy.allocate(oos_df, data_title='')
    opt_policy.evaluate(alloc_eva_df, oos_df, data_title='Evaluate PT data')
    opt_policy.print_time_strings_all_steps()

    # --- Reporting ---
    print("Generating report...")
    report = McfOptPolReport(
        mcf=mcf_model_loaded,
        optpol=opt_policy,
        outputfile=os.path.join(output_folder, "Mcf_Optimal_Policy_Report")
    )
    report.report()
    print("Report generated.")
    print("\nEnd of computations.")


if __name__ == "__main__":
    main()
