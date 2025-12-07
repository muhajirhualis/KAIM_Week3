import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11})

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")


class EDAPipeline:
    def __init__(self, df):
        """
        Initializes the EDA Pipeline with a pandas DataFrame.
        """
        self.df = df
        
    def summarize_data(self):
        """
        Returns descriptive statistics and checks for missing values.
        """
        summary = {
            "description": self.df.describe().T,
            "missing_values": self.df.isnull().sum(),
            "dtypes": self.df.dtypes
            }
        return summary
      
    def plot_distributions(self, numerical_cols, categorical_cols):
        """
        Plots histograms for numerical cols and bar charts for categorical cols.
        """
        # Numerical columns
        for col in numerical_cols:
            if col in self.df.columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(self.df[col], kde=True, bins=30)
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.show()

        # Categorical columns
        for col in categorical_cols:
            if col in self.df.columns:
                plt.figure(figsize=(10, 6))
                self.df[col].value_counts().plot(kind='bar')
                plt.title(f'Count of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.show()


    def plot_log_scaled_distributions(self, numerical_cols):
        plt.figure(figsize=(10, 6))
        
        # Seaborn automatically handles the log transformation and binning
        # The `log_scale=True` argument applies log transformation to the x-axis
        
        for col in numerical_cols:
            if col in self.df.columns:
                sns.histplot(
                    data=self.df[self.df[col] > 0], # Filter out zero values
                    x=col, 
                    kde=True, 
                    bins=50, 
                    log_scale=True 
                )
                
                plt.title(f'Log-Scaled Distribution of {col}')
                # Seaborn automatically updates the axis label to show log scale
                plt.show()
         
    def bivariate_analysis(self):
        """
        Correlations and Scatter plots for Premium vs Claims by ZipCode.
        """
        # Scatter plot: TotalPremium vs TotalClaims
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, x='TotalPremium', y='TotalClaims', alpha=0.5)
        plt.title('TotalPremium vs TotalClaims')
        plt.show()

        # Correlation Matrix for numerical columns
        numeric_df = self.df.select_dtypes(include=[np.number])
        plt.figure(figsize=(12, 10))
        sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()
        
        # MONTHLY TRENDS BY ZIPCODE 
               
        # 1. Aggregate data by month and zip code
        monthly_zip_data = self.df.groupby('PostalCode').agg(
            TotalPremium_Monthly=('TotalPremium', 'sum'),
            TotalClaims_Monthly=('TotalClaims', 'sum')
        ).reset_index()

        # 2. Correlation Matrix
        correlation = monthly_zip_data[['TotalPremium_Monthly', 'TotalClaims_Monthly']].corr()
        print("\nCorrelation between Monthly Premium and Claims (Aggregated by ZipCode):\n", correlation)

        # 3. Visualization (Scatter Plot)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=monthly_zip_data, 
            x='TotalPremium_Monthly', 
            y='TotalClaims_Monthly',
            alpha=0.6,
            color='darkblue'
        )
        plt.title('Monthly Claims vs. Premium (Aggregated by ZipCode)')
        plt.xlabel('Monthly Total Premium')
        plt.ylabel('Monthly Total Claims')
        plt.show()
        
        
        
    def compare_trends_over_geography(self, geo_col='Province', metric='TotalClaims'):
        """
        Compare a metric across geographical regions.
        """
        if geo_col in self.df.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=geo_col, y=metric, data=self.df)
            plt.xticks(rotation=45)
            plt.title(f'{metric} distribution by {geo_col}')
            plt.show()
            
    def detect_outliers(self, cols):
        """
        Uses Box Plots to visualize outliers.
        """
        for col in cols:
            if col in self.df.columns:
                plt.figure(figsize=(10, 4))
                sns.boxplot(x=self.df[col])
                plt.title(f'Outliers in {col}')
                plt.show()
                
            
    def loss_ratio(self):
        
        print(f"Sum of Totalclaim : {self.df['TotalClaims'].sum()}")
        print(f"Sum of TotalPremium : {self.df['TotalPremium'].sum()}")
        
        calculate_loss_ratio = self.df['TotalClaims'].sum() / self.df['TotalPremium'].sum()
        print(f"Loss Ratio = {calculate_loss_ratio}") 
        
    
    def visuaize_loss_ratio(self):
        """Visualizes how Loss Ratio varies by Province, VehicleType, and Gender."""
        # Helper for clean plotting
        def plot_lr_segmented(group_col, title):
            if group_col not in self.df.columns:
                return
            lr_df = self.df.groupby(group_col)[['TotalClaims', 'TotalPremium']].sum()
            lr_df['LossRatio'] = lr_df['TotalClaims'] / lr_df['TotalPremium']
            lr_df = lr_df.sort_values('LossRatio').dropna()

            plt.figure(figsize=(10, 6))
            colors = ['crimson' if x > 1 else 'forestgreen' for x in lr_df['LossRatio']]
            lr_df['LossRatio'].plot(kind='barh', color=colors, edgecolor='black', alpha=0.85)
            plt.axvline(1.0, color='black', linestyle='--', linewidth=1.2, label='Break-even (LR=1.0)')
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel('Loss Ratio')
            plt.legend()
            plt.grid(axis='x', linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()

        plot_lr_segmented('Province', 'Loss Ratio by Province')
        plot_lr_segmented('VehicleType', 'Loss Ratio by Vehicle Type')
        plot_lr_segmented('Gender', 'Loss Ratio by Gender')
    
    def temporal_trends(self):
        """
        Analyze how claim frequency and severity changed over the 18-month period.
        """
        if 'TransactionMonth' not in self.df.columns:
            print(" 'TransactionMonth' column missing. Skipping temporal analysis.")
            return

        # Focus only on rows with claims for severity
        claims_only = self.df[self.df['TotalClaims'] > 0]

        # Aggregate by month
        monthly_summary = self.df.groupby('TransactionMonth').agg(
        TotalPolicies=('PolicyID', 'count'),
        TotalClaimsSum=('TotalClaims', 'sum')
        ).reset_index()

        monthly_severity = claims_only.groupby('TransactionMonth').agg(
        ClaimCount=('PolicyID', 'count'),
        AvgClaimAmount=('TotalClaims', 'mean')
        ).reset_index()

        # Merge to align months
        merged = monthly_summary.merge(monthly_severity, on='TransactionMonth', how='left')
        merged['ClaimFrequency'] = merged['ClaimCount'] / merged['TotalPolicies']

        # Plot: Dual-axis chart
        fig, ax1 = plt.subplots(figsize=(12, 6))
        color1, color2 = 'steelblue', 'crimson'

        ax1.set_xlabel('Transaction Month')
        ax1.set_ylabel('Avg Claim Amount (Severity)', color=color1)
        ax1.plot(merged['TransactionMonth'], merged['AvgClaimAmount'], 
                color=color1, marker='o', label='Avg Claim Amount')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, which='major', axis='x', linestyle='--', alpha=0.5)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Claim Frequency', color=color2)
        ax2.plot(merged['TransactionMonth'], merged['ClaimFrequency'], 
                color=color2, marker='s', linestyle='--', label='Claim Frequency')
        ax2.tick_params(axis='y', labelcolor=color2)

        plt.title('Claim Severity & Frequency Over Time (2014–2015)')
        fig.tight_layout()
        plt.show()

        # Print key insight
        trend_severity = np.polyfit(range(len(merged)), merged['AvgClaimAmount'].fillna(0), 1)[0]
        trend_freq = np.polyfit(range(len(merged)), merged['ClaimFrequency'].fillna(0), 1)[0]
        print(f"Trend Insight: Severity {'↑ increasing' if trend_severity > 0 else '↓ decreasing'}," 
            f" Frequency {'↑ increasing' if trend_freq > 0 else '↓ decreasing'} over time.")
        
    def vehicle_claim_amounts(self, top_n: int = 10):
            """
            Identifies vehicle makes/models associated with the highest and lowest average claim amounts.
            """
            
            if 'Model' not in self.df.columns:
                print("Error: 'Model' column is required for this analysis.")
                return

            # Focus only on policies that actually had a claim
            claims_only_df = self.df[self.df['TotalClaims'] > 0]
            
            # Calculate Average Claim Amount and Policy Count per Model
            vehicle_risk = claims_only_df.groupby('Model').agg(
                AvgClaimAmount=('TotalClaims', 'mean'),
                ClaimCount=('TotalClaims', 'count')
            ).reset_index()
            
            # Filter out Models with very low claim counts (e.g., < 10) to ensure stability
            vehicle_risk = vehicle_risk[vehicle_risk['ClaimCount'] >= 10] 
            
            # Sort for Highest and Lowest
            highest_claims = vehicle_risk.sort_values(by='AvgClaimAmount', ascending=False).head(top_n)
            lowest_claims = vehicle_risk.sort_values(by='AvgClaimAmount', ascending=True).head(top_n)

            print(f"\nTop {top_n} Vehicle Models by HIGHEST Average Claim Amount:\n", highest_claims)
            print(f"\nTop {top_n} Vehicle Models by LOWEST Average Claim Amount:\n", lowest_claims)

            # Visualization: Combined Top/Bottom N
            plot_data = pd.concat([highest_claims, lowest_claims])
            
            plt.figure(figsize=(12, 8))
            sns.barplot(
                x='AvgClaimAmount', 
                y='Model', 
                data=plot_data, 
                palette='coolwarm',
                order=plot_data.sort_values(by='AvgClaimAmount', ascending=False)['Model']
            )
            plt.title(f'Vehicle Model Claim Severity (Average Claim Amount) - Top/Bottom {top_n}')
            plt.xlabel('Average Claim Amount')
            plt.ylabel('Vehicle Model')
            plt.show()
                
