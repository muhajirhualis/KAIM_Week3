
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
from typing import Dict, Any, List

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

class HypothesisTester:
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the tester with the clean DataFrame and calculates the Margin metric.
        """
        self.df = df.copy()
        # Margin (Profit) is a critical metric for H0 testing
        self.df['Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']

    def _report_result(self, hypothesis: str, p_value: float, test_stat: float, test_name: str) -> Dict[str, Any]:
        """Formats the statistical test output."""
        
        is_significant = p_value < 0.05
        decision = "REJECT Null Hypothesis" if is_significant else "FAIL TO REJECT Null Hypothesis"
        
        print(f"\n--- {hypothesis} ({test_name}) ---")
        print(f"P-Value: {p_value:.4f}")
        print(f"Test Statistic: {test_stat:.4f}")
        print(f"Decision: {decision} (Alpha=0.05)")
        
        return {
            'hypothesis': hypothesis,
            'test_name': test_name,
            'p_value': p_value,
            'decision': decision
        }

    # 1. TEST FREQUENCY DIFFERENCES (Claim Rate, HasClaim) - CHI-SQUARED
    # =================================================================
    def test_frequency_difference(self, group_col: str, min_group_count: int = 100):
        """
        Tests the Null Hypothesis (H0) that 'Claim Frequency' (HasClaim rate) 
        is the same across different categories of a grouping column (e.g., Province).
        Uses the Chi-Squared Test.
        """
        hypothesis = f"H0: Claim Frequency is the same across all groups in '{group_col}'"
        
        # Filter groups to ensure statistical reliability
        counts = self.df[group_col].value_counts()
        valid_groups = counts[counts >= min_group_count].index
        test_df = self.df[self.df[group_col].isin(valid_groups)].copy()

        if len(valid_groups) < 2:
            return self._report_result(hypothesis, 1.0, 0.0, "Chi-Squared - Insufficient Groups")

        # Create a contingency table: Group vs. HasClaim (0/1)
        contingency_table = pd.crosstab(test_df[group_col], test_df['HasClaim'])
        
        # Execute the Chi-Squared Test
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        
        return self._report_result(hypothesis, p_value, chi2, "Chi-Squared Test")


    # 2. TEST MEAN DIFFERENCE: TWO GROUPS (e.g., Men vs. Women) - T-TEST
    # =================================================================
    def test_mean_difference_two_groups(self, group_col: str, metric_col: str, group_a: str, group_b: str):
        """
        Tests the H0 that the Mean of a metric (e.g., Claim Severity or Margin)
        is the same between exactly two specific groups (A vs. B). Uses T-Test.
        """
        
        # Adjust hypothesis and data filter based on the metric
        if metric_col == 'TotalClaims':
            hypothesis = f"H0: Claim Severity is the same between {group_a} and {group_b}"
            # Only test policies that had a claim for Severity analysis
            test_df = self.df[self.df['HasClaim'] == 1].copy()
        else:
            hypothesis = f"H0: Mean '{metric_col}' is the same between {group_a} and {group_b}"
            test_df = self.df.copy()

        # Extract data for the two groups
        data_a = test_df[test_df[group_col] == group_a][metric_col].values
        data_b = test_df[test_df[group_col] == group_b][metric_col].values
        
        # Check for sufficient data
        if len(data_a) < 2 or len(data_b) < 2:
            return self._report_result(hypothesis, 1.0, 0.0, "T-Test - Insufficient Samples")

        # Execute Welch's T-Test (safer, assumes unequal variance)
        t_stat, p_value = ttest_ind(data_a, data_b, equal_var=False)
        
        return self._report_result(hypothesis, p_value, abs(t_stat), "Two-Sample T-Test")


    # =================================================================
    # 3. TEST MEAN DIFFERENCE: MULTIPLE GROUPS (e.g., Provinces) - ANOVA
    # =================================================================
    def test_mean_difference_multiple_groups(self, group_col: str, metric_col: str, min_group_count: int = 100):
        """
        Tests the H0 that the Mean of a metric (e.g., Claim Severity or Margin)
        is the same across more than two categories. Uses One-Way ANOVA.
        """
        
        # Adjust hypothesis and data filter based on the metric
        if metric_col == 'TotalClaims':
            hypothesis = f"H0: Claim Severity is the same across all groups in '{group_col}'"
            # Only test policies that had a claim for Severity analysis
            test_df = self.df[self.df['HasClaim'] == 1].copy()
        else:
            hypothesis = f"H0: Mean '{metric_col}' is the same across all groups in '{group_col}'"
            test_df = self.df.copy()

        # Filter groups to ensure statistical reliability
        counts = test_df[group_col].value_counts()
        valid_groups = counts[counts >= min_group_count].index
        
        if len(valid_groups) < 2:
            return self._report_result(hypothesis, 1.0, 0.0, "ANOVA - Insufficient Groups")

        # Prepare list of arrays for ANOVA test
        groups_data = [
            test_df[test_df[group_col] == group][metric_col].values 
            for group in valid_groups
        ]
        
        # Execute One-Way ANOVA
        f_stat, p_value = f_oneway(*groups_data)
        
        return self._report_result(hypothesis, p_value, f_stat, "One-Way ANOVA")