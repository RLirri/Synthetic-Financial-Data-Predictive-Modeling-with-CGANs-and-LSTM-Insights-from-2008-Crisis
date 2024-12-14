import torch
from CGAN2 import ConditionalGenerator
from torchviz import make_dot

# Load the model
generator = ConditionalGenerator(z_dim=100, condition_dim=2, output_dim=5, seq_length=30)
generator.load_state_dict(torch.load("improved_conditional_generator_model.pth"))

# Create dummy input tensors
z = torch.randn(1, 30, 100)  # Noise
conditions = torch.randn(1, 30, 2)  # Conditions

# Forward pass to visualize
output = generator(z, conditions)

# Generate a computation graph
dot = make_dot(output, params=dict(list(generator.named_parameters())))

# Save the graph
dot.render("generator_architecture", format="png")
