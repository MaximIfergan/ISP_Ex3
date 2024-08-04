import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Parse the string representation of the list into a Python list
with open('results', "r") as f:
    data_str = f.read()
data = ast.literal_eval(data_str)

# Convert the data into a pandas DataFrame
df = pd.DataFrame([({key : val for key,val in ast.literal_eval(item[0][:-1].strip('dict_items('))}, item[1]) for item in data], columns=['params', 'accuracy'])

# Extract individual parameters
df['window_size'] = df['params'].apply(lambda x: x['window_size'])
df['optimizer_name'] = df['params'].apply(lambda x: x['optimizer_name'])
df['lr'] = df['params'].apply(lambda x: x['optimizer_params']['lr'])
df['model_name'] = df['params'].apply(lambda x: x['model_name'])
df['hidden_size'] = df['params'].apply(lambda x: x['hidden_size'])
df['batch_size'] = df['params'].apply(lambda x: x['batch_size'])
df['num_epochs'] = df['params'].apply(lambda x: x['num_epochs'])


# Function to plot parameter comparison
def plot_parameter_comparison(df, param, title):
    plt.figure(figsize=(12, 6))

    # Calculate mean and std for each parameter value
    summary = df.groupby(param)['accuracy'].agg(['mean', 'std']).reset_index()

    # Sort the summary DataFrame by the parameter values
    summary = summary.sort_values(param)

    # Create x-positions for the bars
    x_pos = np.arange(len(summary))

    # Plot bars
    plt.bar(x_pos, summary['mean'], yerr=summary['std'], capsize=5,
            color='skyblue', edgecolor='navy', alpha=0.8)

    # Customize the plot
    plt.title(f'Average Validation Accuracy by {title} Comparison', fontsize=16)
    plt.ylabel('Average Validation Accuracy', fontsize=12)
    plt.xlabel(title, fontsize=12)

    # Set y-axis limit from 0 to slightly above the maximum value
    max_value = summary['mean'].max() + summary['std'].max()
    plt.ylim(0, min(1, max_value * 1.1))

    # Highlight the best model
    best_model = df.loc[df['accuracy'].idxmax()]
    best_param_value = best_model[param]
    best_accuracy = best_model['accuracy']

    # Find the index of the best model in the summary DataFrame
    best_index = summary[summary[param] == best_param_value].index[0]

    plt.plot(best_index, best_accuracy, 'r*', markersize=20, label='Best Model')

    # Set x-ticks and labels
    plt.xticks(x_pos, summary[param])

    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(f'{param}_comparison.png')

    # Show the plot
    plt.show()


def plot_parameter_effect(df):
    parameters = ['window_size', 'optimizer_name', 'lr', 'model_name', 'hidden_size', 'batch_size', 'num_epochs']
    titles = ['Window Size', 'Optimizer', 'Learning Rate', 'Model', 'Hidden Size', 'Batch Size', 'Number of Epochs']

    plt.figure(figsize=(15, 10))

    for i, (param, title) in enumerate(zip(parameters, titles), 1):
        plt.subplot(3, 3, i)

        # Calculate mean accuracy for each parameter value
        summary = df.groupby(param)['accuracy'].mean().reset_index()
        summary = summary.sort_values(param)

        # Plot line
        plt.plot(summary[param], summary['accuracy'], marker='o')

        # Customize the plot
        plt.title(title, fontsize=12)
        plt.ylabel('Accuracy' if i % 3 == 1 else '')
        plt.xlabel(title)

        # Set y-axis limit from min to max
        plt.ylim(summary['accuracy'].min() * 0.95, summary['accuracy'].max() * 1.05)

        # Rotate x-axis labels if there are many or if they are strings
        if len(summary) > 5 or summary[param].dtype == 'object':
            plt.xticks(rotation=45, ha='right')

        plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('parameter_effects.png')
    plt.show()


def plot_parameter_importance(df):
    # Prepare the data
    X = df[['window_size', 'optimizer_name', 'lr', 'model_name', 'hidden_size', 'batch_size', 'num_epochs']]
    y = df['accuracy']

    # Encode categorical variables
    le = LabelEncoder()
    X['optimizer_name'] = le.fit_transform(X['optimizer_name'])
    X['model_name'] = le.fit_transform(X['model_name'])

    # Convert learning rate to numeric (assuming it's stored as string)
    X['lr'] = X['lr'].astype(float)

    # Train a Random Forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Get feature importances
    importances = rf.feature_importances_
    feature_names = X.columns

    # Sort features by importance
    indices = np.argsort(importances)[::-1]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.title("Parameter Importance for Accuracy")
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Parameters')
    plt.ylabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('parameter_importance.png')
    plt.show()

    # Print importance scores
    print("\nParameter Importance Scores:")
    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")


plot_parameter_importance(df)

# plot_parameter_effect(df)

# Plot comparisons for each parameter
# plot_parameter_comparison(df, 'window_size', 'Window Size')
# plot_parameter_comparison(df, 'optimizer_name', 'Optimizer')
# plot_parameter_comparison(df, 'lr', 'Learning Rate')
# plot_parameter_comparison(df, 'model_name', 'Model')
# plot_parameter_comparison(df, 'hidden_size', 'Hidden Size')
# plot_parameter_comparison(df, 'batch_size', 'Batch Size')
# plot_parameter_comparison(df, 'num_epochs', 'Number of Epochs')