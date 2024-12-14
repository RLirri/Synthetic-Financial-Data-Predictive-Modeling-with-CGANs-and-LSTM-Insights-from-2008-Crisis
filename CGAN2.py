import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# Download S&P 500 data for 2008 financial crisis (2007-2009)
ticker = '^GSPC'
data = yf.download(ticker, start='2007-01-01', end='2009-12-31')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Generate financial indicators as conditions
data['MA_10'] = data['Close'].rolling(window=10).mean()
data['MA_30'] = data['Close'].rolling(window=30).mean()
data = data.dropna()

# Normalize the data to [0, 1]
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)

# Function to create sequences and align conditions
def create_sequences_with_conditions(data, seq_length):
    sequences, conditions = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length, :5])  # Features: Open, High, Low, Close, Volume
        conditions.append(data[i:i + seq_length, 5:])  # Conditions: MA_10, MA_30
    return np.array(sequences), np.array(conditions)

seq_length = 30
sequences, conditions = create_sequences_with_conditions(normalized_data, seq_length)
sequences_tensor = torch.FloatTensor(sequences)
conditions_tensor = torch.FloatTensor(conditions)

# Conditional GAN Generator with Attention Mechanism
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

# Conditional Discriminator
class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super(ConditionalDiscriminator, self).__init__()
        self.lstm = nn.LSTM(input_dim + condition_dim, 512, num_layers=2, batch_first=True)
        self.fc = nn.Linear(512, 1)

    def forward(self, x, conditions):
        x = torch.cat((x, conditions), dim=-1)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Gradient Penalty Function
def calc_gradient_penalty(D, real_data, fake_data, conditions):
    alpha = torch.rand(real_data.size(0), 1, 1).to(real_data.device)
    alpha = alpha.expand_as(real_data)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.requires_grad_(True)

    d_interpolates = D(interpolates, conditions)
    fake = torch.ones(real_data.size(0), 1).to(real_data.device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Hyperparameters
z_dim = 100
input_dim = sequences.shape[2]
output_dim = input_dim
condition_dim = conditions.shape[2]
batch_size = 128
learning_rate = 0.0001
n_epochs = 1000

# Models
generator = ConditionalGenerator(z_dim, condition_dim, output_dim, seq_length)
discriminator = ConditionalDiscriminator(output_dim, condition_dim)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)
sequences_tensor, conditions_tensor = sequences_tensor.to(device), conditions_tensor.to(device)

for epoch in range(n_epochs):
    for i in range(0, len(sequences_tensor), batch_size):
        real_data_batch = sequences_tensor[i:i + batch_size]
        conditions_batch = conditions_tensor[i:i + batch_size]
        batch_real_size = real_data_batch.size(0)

        # Generate noise and synthetic data
        z = torch.randn(batch_real_size, seq_length, z_dim).to(device)
        fake_data = generator(z, conditions_batch)

        # Train discriminator
        optimizer_D.zero_grad()
        real_loss = discriminator(real_data_batch, conditions_batch).mean()
        fake_loss = discriminator(fake_data, conditions_batch).mean()
        gradient_penalty = calc_gradient_penalty(discriminator, real_data_batch, fake_data, conditions_batch)
        d_loss = fake_loss - real_loss + 10 * gradient_penalty
        d_loss.backward()
        optimizer_D.step()

        # Train generator with feature matching loss
        optimizer_G.zero_grad()
        fake_data = generator(z, conditions_batch)
        real_features = discriminator(real_data_batch, conditions_batch)
        fake_features = discriminator(fake_data, conditions_batch)
        feature_loss = torch.mean((real_features - fake_features) ** 2)
        g_loss = -torch.mean(discriminator(fake_data, conditions_batch)) + feature_loss
        g_loss.backward()
        optimizer_G.step()

    print(f'Epoch [{epoch+1}/{n_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

# Save the trained generator
torch.save(generator.state_dict(), "improved_conditional_generator_model.pth")
print("Improved Conditional Generator model saved as 'improved_conditional_generator_model.pth'")
