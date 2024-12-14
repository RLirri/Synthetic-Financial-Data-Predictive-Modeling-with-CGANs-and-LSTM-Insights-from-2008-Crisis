import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# Define ConditionalGenerator
class ConditionalGenerator(nn.Module):
    def __init__(self, z_dim, condition_dim, output_dim, seq_length):
        super(ConditionalGenerator, self).__init__()
        self.seq_length = seq_length
        self.output_dim = output_dim
        self.lstm = nn.LSTM(z_dim + condition_dim, 512, num_layers=2, batch_first=True)
        self.attention = nn.Linear(512, 1)
        self.fc = nn.Linear(512, self.seq_length * self.output_dim)

    def forward(self, z, conditions):
        z = torch.cat((z, conditions), dim=-1)
        lstm_out, _ = self.lstm(z)
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        x = self.fc(context_vector)
        x = x.view(-1, self.seq_length, self.output_dim)
        return x

# Parameters (must match training script)
z_dim = 100
seq_length = 30
output_dim = 5  # Features: Open, High, Low, Close, Volume
condition_dim = 2  # Conditions: MA_10, MA_30

# Load generator model
generator = ConditionalGenerator(z_dim, condition_dim, output_dim, seq_length)
generator.load_state_dict(torch.load("improved_conditional_generator_model.pth"))
generator.eval()

# Load real data and separate features and conditions
real_data = pd.read_csv("real_data.csv")  # Ensure this is the real data with features and conditions
feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
condition_columns = ['MA_10', 'MA_30']

features = real_data[feature_columns]
conditions = real_data[condition_columns]

# Normalize features and conditions separately
feature_scaler = MinMaxScaler()
condition_scaler = MinMaxScaler()

normalized_features = feature_scaler.fit_transform(features)
normalized_conditions = condition_scaler.fit_transform(conditions)

# Generate synthetic data
batch_size = 128  # Number of samples to generate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)

z = torch.randn(batch_size, seq_length, z_dim, dtype=torch.float32).to(device)
conditions_tensor = torch.tensor(normalized_conditions[:batch_size, :], dtype=torch.float32)
conditions_tensor = conditions_tensor.unsqueeze(1).expand(-1, seq_length, -1).to(device)

synthetic_data = generator(z, conditions_tensor).detach().cpu().numpy()

# Rescale synthetic data back to original scale
synthetic_data_rescaled = feature_scaler.inverse_transform(
    synthetic_data.reshape(-1, synthetic_data.shape[2])
)

# Save synthetic data
synthetic_data_df = pd.DataFrame(synthetic_data_rescaled, columns=feature_columns)
synthetic_data_df.to_csv("synthetic_data_conditional.csv", index=False)

print("Synthetic data saved as 'synthetic_data_conditional.csv'.")

# Comparison and Validation
# Perform statistical tests, feature distribution comparison, and correlation checks
ks_results = {}

for column in feature_columns:  # Compare only feature columns
    real_values = real_data[column][:synthetic_data_rescaled.shape[0]]
    synthetic_values = synthetic_data_df[column]
    ks_stat, p_value = ks_2samp(real_values, synthetic_values)
    ks_results[column] = {'KS Statistic': ks_stat, 'P-Value': p_value}

ks_results_df = pd.DataFrame.from_dict(ks_results, orient='index')
ks_results_df.to_csv("ks_test_results_conditional.csv")

print("KS test results saved as 'ks_test_results_conditional.csv'.")

# Plot real vs. synthetic distributions for validation
for column in feature_columns:
    plt.figure(figsize=(10, 6))
    plt.hist(real_data[column], bins=50, alpha=0.7, label="Real Data", color='blue')
    plt.hist(synthetic_data_df[column], bins=50, alpha=0.7, label="Synthetic Data", color='orange')
    plt.title(f"Feature Distribution: {column}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
