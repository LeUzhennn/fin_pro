import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

from config import DEFAULT_DATA_FILE, PLOTS_DIR, DataConfig
from src.logger import get_logger

logger = get_logger(__name__)

# 從配置取得設定
DATA_FILE = str(DEFAULT_DATA_FILE)
SELECTED_FEATURES = DataConfig.SELECTED_FEATURES_FOR_VISUALIZATION

def clean_and_load_data(file_path):
    """Loads and cleans the dataset."""
    df = pd.read_csv(file_path, low_memory=False)

    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()

    # Attempt to convert relevant columns to numeric, handling errors
    for col in SELECTED_FEATURES + ['Flow Byts/s', 'Flow Pkts/s']: # Include other known problematic numeric cols
        if col in df.columns:
            # Replace ',' with '' and then convert to numeric
            df[col] = df[col].astype(str).str.replace(',', '').replace('Infinity', np.nan).replace('-Infinity', np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill any remaining NaNs after conversion for relevant features
    # Or decide to drop rows with NaNs in critical columns, depending on analysis goal
    for col in SELECTED_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median()) # Filling with median as a simple strategy

    # Ensure 'Label' column is stripped of whitespace
    if 'Label' in df.columns:
        df['Label'] = df['Label'].str.strip()

    return df

def visualize_features(df, features, plots_dir):
    """Generates and saves visualizations for selected features."""
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    labels = df['Label'].unique()
    max_labels = DataConfig.MAX_LABELS_FOR_PLOT
    if len(labels) > max_labels:
        logger.warning(f"標籤過多 ({len(labels)})，僅視覺化前 {max_labels-1} 個和 'Benign'")
        print(f"Warning: Too many unique labels ({len(labels)}). Visualizing top {max_labels-1} and 'Benign'.")
        top_labels = df['Label'].value_counts().nlargest(4).index.tolist()
        if 'Benign' not in top_labels:
            top_labels.append('Benign')
        df_filtered = df[df['Label'].isin(top_labels)].copy()
        labels_to_plot = df_filtered['Label'].unique()
    else:
        df_filtered = df.copy()
        labels_to_plot = labels

    print(f"Visualizing for labels: {labels_to_plot}")

    for feature in features:
        if feature not in df_filtered.columns:
            print(f"Feature '{feature}' not found in DataFrame. Skipping.")
            continue

        # KDE Plot
        plt.figure(figsize=(12, 6))
        for label in labels_to_plot:
            subset = df_filtered[df_filtered['Label'] == label]
            if not subset.empty:
                sns.kdeplot(subset[feature], label=label, fill=True)
        plt.title(f'KDE Plot of {feature} by Label')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        plt.xscale('log') # Use log scale for features that can have large ranges
        plt.savefig(os.path.join(plots_dir, f'{feature}_kde_plot.png'))
        plt.close()
        print(f"Saved {feature}_kde_plot.png")

        # Box Plot
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Label', y=feature, data=df_filtered, palette='viridis')
        plt.title(f'Box Plot of {feature} by Label')
        plt.xlabel('Label')
        plt.ylabel(feature)
        plt.xticks(rotation=45, ha='right')
        plt.yscale('log') # Use log scale for features that can have large ranges
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{feature}_boxplot.png'))
        plt.close()
        print(f"Saved {feature}_boxplot.png")

def main():
    print(f"Loading data from {DATA_FILE}...")
    df = clean_and_load_data(DATA_FILE)
    print(f"Data loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    print(f"Unique labels: {df['Label'].unique()}")

    visualize_features(df, SELECTED_FEATURES, PLOTS_DIR)
    print("EDA visualizations complete. Check the 'plots' directory.")

if __name__ == "__main__":
    main()
