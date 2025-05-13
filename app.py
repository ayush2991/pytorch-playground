import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import time
import pandas as pd
import altair as alt
from typing import Tuple, List

# Define a simple model for binary classification
class BinaryClassifier(nn.Module):
    """A simple neural network for binary classification tasks.
    
    This model consists of two fully connected layers with ReLU activation
    and sigmoid output for binary classification.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """Initialize the binary classifier.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in the hidden layer
            output_size: Number of output classes (1 for binary classification)
        """
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size) with probabilities
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
            
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  # Output probability between 0 and 1
        return x

# Streamlit app
st.title("A Simple Neural Net Learning to Classify")

# Default values for hyperparameters
DEFAULT_VALUES = {
    "num_samples": 200,
    "train_test_split": 0.5,
    "hidden_size": 10,
    "learning_rate": 0.01,
    "epochs": 50,
    "batch_size": 16,
    "noise_std": 0.2
}

# Initialize session state for hyperparameters if not exists
if "initialized" not in st.session_state:
    for key, value in DEFAULT_VALUES.items():
        st.session_state[key] = value
    st.session_state.initialized = True

# Hyperparameters
input_size = 2  # 2D input data
st.sidebar.header("Hyperparameters")
reset_button = st.sidebar.button("Reset")

# Reset values when button is clicked
if reset_button:
    for key, value in DEFAULT_VALUES.items():
        st.session_state[key] = value
    st.rerun()  # Rerun the app to update all widgets

# Sliders with session state
num_samples = st.sidebar.slider("Number of Samples", 100, 400, key="num_samples")
train_test_split = st.sidebar.slider("Train-Test Split", 0.1, 0.9, key="train_test_split")
hidden_size = st.sidebar.slider("Hidden Size", 5, 20, key="hidden_size")
output_size = 1  # Binary classification: probability of class 1
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, key="learning_rate")
epochs = st.sidebar.slider("Epochs", 1, 200, key="epochs")
batch_size = st.sidebar.slider("Batch Size", 1, 32, key="batch_size")
noise_std = st.sidebar.slider("Noise Standard Deviation", 0.01, 1.0, key="noise_std")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Generate 2D data with labels based on distance from origin
X_np = np.random.randn(num_samples, 2)
distances = np.linalg.norm(X_np, axis=1)
y_np = (distances < 1).astype(np.float32).reshape(-1, 1)  # Label 1 if distance < 1, else 0
# Add noise to the data
y_np += np.random.normal(0, noise_std, y_np.shape)
# Make labels binary
y_np = (y_np > 0.5).astype(np.float32)  # Convert to binary labels

# Convert to PyTorch tensors and create dataset
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)
dataset = TensorDataset(X, y)

# Split into train and test sets (80-20 split)
train_size = int(train_test_split * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Get test data for visualization
test_X = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
test_y = torch.stack([test_dataset[i][1] for i in range(len(test_dataset))])
test_X_np = test_X.numpy()
test_y_np = test_y.numpy()

# --- Display initial data scatterplot ---
st.subheader("Training and Test Data")
# Create DataFrame for visualization
train_indices = train_dataset.indices
test_indices = test_dataset.indices

df_train = pd.DataFrame({
    'X1': X_np[train_indices, 0],
    'X2': X_np[train_indices, 1],
    'Label': y_np[train_indices, 0],
    'Set': 'Train'
})

df_test = pd.DataFrame({
    'X1': X_np[test_indices, 0],
    'X2': X_np[test_indices, 1],
    'Label': y_np[test_indices, 0],
    'Set': 'Test'
})

df = pd.concat([df_train, df_test])

chart = (
    alt.Chart(df)
    .mark_circle()
    .encode(
        x="X1",
        y="X2",
        color=alt.Color("Label:N", scale=alt.Scale(range=["red", "blue"])),
        shape=alt.Shape("Set:N", scale=alt.Scale(range=["circle", "square"])),
        tooltip=["X1", "X2", "Label", "Set"],
    )
    .properties(title="Training and Test Data Points")
)
st.altair_chart(chart, use_container_width=True)
# --- End of initial data scatterplot ---

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BinaryClassifier(input_size, hidden_size, output_size).to(device)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Placeholders for the loss plot and data scatter plot during training
loss_chart = st.empty()
scatter_chart = st.empty()
train_losses: List[float] = []
val_losses: List[float] = []
epochs_list: List[int] = []

if st.button("Start Training"):
    st.write("Starting Training...")
    progress_bar = st.progress(0)
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item()
        
        epoch_val_loss = running_val_loss / len(test_loader)
        val_losses.append(epoch_val_loss)
        epochs_list.append(epoch + 1)
        
        # Update progress bar
        progress_bar.progress((epoch + 1) / epochs)

        # Create the loss plot using streamlit
        loss_df = pd.DataFrame({
            "Epoch": epochs_list + epochs_list,
            "Loss": train_losses + val_losses,
            "Type": ["Training"] * len(train_losses) + ["Validation"] * len(val_losses)
        })
        
        chart = (
            alt.Chart(loss_df)
            .mark_line()
            .encode(
                x="Epoch",
                y="Loss",
                color="Type",
                tooltip=["Epoch", "Loss", "Type"]
            )
            .properties(title="Training and Validation Loss per Epoch")
        )
        loss_chart.altair_chart(chart, use_container_width=True)

        # Create the scatter plot of the test data with predicted probabilities
        with torch.no_grad():
            predicted_probs = model(test_X.to(device)).cpu().numpy()

        df_scatter = pd.DataFrame(
            np.hstack((test_X_np, predicted_probs, test_y_np)),
            columns=["X1", "X2", "Predicted Probability", "Label"],
        )

        scatter_chart.altair_chart(
            alt.Chart(df_scatter)
            .mark_circle()
            .encode(
                x="X1",
                y="X2",
                color=alt.Color(
                    "Predicted Probability", scale=alt.Scale(scheme="viridis")
                ),
                tooltip=["X1", "X2", "Predicted Probability", "Label"],
            )
            .properties(
                title=f"Epoch {epoch+1}: Test Data Points with Predicted Probabilities"
            ),
            use_container_width=True,
        )

        time.sleep(0.1)

    st.write("Training Finished!")