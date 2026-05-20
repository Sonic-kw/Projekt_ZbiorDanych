import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.utils.logger import setup_logger
from sklearn.linear_model import LinearRegression

logger = setup_logger("statistics")

class MarketAnalysis:
    @staticmethod
    def calculate_market_shares(df: pd.DataFrame, column: str = "brand") -> pd.Series:
        """Calculates the percentage share of each brand."""
        shares = df[column].value_counts(normalize=True) * 100
        return shares

    @staticmethod
    def plot_market_structure(df: pd.DataFrame, output_path: str = "results/market_structure.png"):
        """Plots the market share of brands as a horizontal bar chart."""
        shares = MarketAnalysis.calculate_market_shares(df)
        plt.figure(figsize=(12, 8))
        sns.barplot(x=shares.values, y=shares.index, palette="viridis")
        plt.title("Market Share by Brand (%)", fontsize=15)
        plt.xlabel("Share (%)")
        plt.ylabel("Brand")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Market structure plot saved to {output_path}")

    @staticmethod
    def plot_distributions(df: pd.DataFrame, output_path: str = "results/distributions.png"):
        """Plots the distributions of price and mileage using histograms and KDE."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Price Distribution
        sns.histplot(df['price'], kde=True, ax=axes[0], color='blue')
        axes[0].set_title("Price Distribution", fontsize=13)
        axes[0].set_xlabel("Price (PLN)")
        axes[0].set_ylabel("Frequency")
        
        # Mileage Distribution
        sns.histplot(df['mileage'], kde=True, ax=axes[1], color='green')
        axes[1].set_title("Mileage Distribution", fontsize=13)
        axes[1].set_xlabel("Mileage (km)")
        axes[1].set_ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Distributions plot saved to {output_path}")

    @staticmethod
    def plot_regression_analysis(df: pd.DataFrame, x_col: str, y_col: str = 'price', output_path: str = "results/regression.png"):
        """Plots a scatter plot with a linear regression line."""
        plt.figure(figsize=(10, 6))
        sns.regplot(data=df, x=x_col, y=y_col, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        plt.title(f"Relationship: {y_col.capitalize()} vs {x_col.capitalize()}", fontsize=15)
        plt.xlabel(x_col.capitalize())
        plt.ylabel(y_col.capitalize())
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Regression plot for {x_col} saved to {output_path}")

    @staticmethod
    def plot_clusters(df: pd.DataFrame, output_path: str = "results/clusters.png"):
        """Plots K-means clusters (Price vs Mileage)."""
        if 'cluster' not in df.columns:
            logger.error("DataFrame does not contain 'cluster' column. Skipping cluster plot.")
            return

        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x='mileage', y='price', hue='cluster', palette='deep', s=100, alpha=0.7)
        plt.title("Market Segmentation (K-means Clustering)", fontsize=15)
        plt.xlabel("Mileage (km)")
        plt.ylabel("Price (PLN)")
        plt.legend(title="Cluster")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Cluster plot saved to {output_path}")

    @staticmethod
    def plot_correlation_matrix(df: pd.DataFrame, output_path: str = "results/correlation.png"):
        """Plots a heatmap of the correlation matrix."""
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix of Motorcycle Features", fontsize=15)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Correlation matrix saved to {output_path}")
