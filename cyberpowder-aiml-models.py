# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# O-RAN AI/ML Model Development, Training, and Export

Prerequisites:

- Completion of the O-RAN Datasets and Analysis Session and uploading your
  processed dataset to HuggingFace
- Reading chapters 1 and 2 from [Understanding Deep Learning by Simon
  Prince](https://udlbook.github.io/udlbook/) (optional if you are already
  familiar with ML concepts). The other resources on the book website are also
  quite useful.
- Reading this [PyTorch tutorial on Linear
  Regression](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_linear_regression/)
  (optional if you are already somewhat familiar with PyTorch)
- You might want to make a copy of this notebook to your own Google drive
  account so you can save your work. You can do this by clicking on the "Save a
  copy in Drive" in the File menu.
- **Note: Don't just run the whole notebook. There are some cells that will
  require interaction. Read the comments and understand what each cell does.**
"""

# %%
# Install required packages (various other required packages are already available in the colab environment)
# !uv -q pip install datasets onnx onnxruntime

# %%
# Import required packages
# Standard library imports
import datetime
import json
import os
import shutil
import tempfile

import datasets
import huggingface_hub as hf
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

# From-style imports
from huggingface_hub import hf_hub_download
from plotly.subplots import make_subplots

# %% [markdown]
"""
## Add Your Token as a Secret in Colab, Log in to HuggingFace, and Download Your Processed Dataset

### Walkthrough adding your token as a secret in Colab

After adding the token as a secret in Colab and use the
following code to get the token:

```
from google.colab import userdata
userdata.get('hf_token')
```
"""

# %% [markdown]
"""
### Log in to HuggingFace
"""
# %%

from google.colab import userdata
token = userdata.get('hf_token')
hf.login(token=token)

# %% [markdown]
"""
### Check that you're logged in.
"""

# %%
username = hf.whoami()['name']
print(f"Logged in as {username}")

# %% [markdown]
"""
### Download your dataset

First, let's list the datasets you have access to under your account.
"""

# %%
hf_api = hf.HfApi()
my_datasets = hf_api.list_datasets(author=username)
for ds in my_datasets:
    print(f"Dataset: {ds.id}")

# %% [markdown]
"""
Now download the processed dataset you uploaded in the previous session. You
should have uploaded it to your personal HuggingFace account, not the
CyberPowder Org.
"""

# %%
dataset_name = "cyberpowder-network-metrics"  # Fill in the name of the processed dataset
dataset_id = f"{username}/{dataset_name}"

# Download the dataset
dataset = datasets.load_dataset(dataset_id)
dataset


# %% [markdown]
"""
### Verify the dataset looks like you expect

You can create a pandas DataFrame from the dataset, then you use the DataFrame
to create a plots to inspect it.
"""

# %%
df = dataset['train'].to_pandas()

# change timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# %%
# Let's regenerate the plots from the previous session
fig = go.Figure()

# Add each metric as a separate trace
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['atten'], mode='lines', name='atten'))
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['CQI'], mode='lines', name='CQI'))
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSRP'], mode='lines', name='RSRP'))
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['DRB.UEThpDl'] / 1000.0, mode='lines', name='DRB.UEThpDl (Mbps)'))
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['min_prb_ratio'], mode='lines', name='min_prb_ratio'))

# Update layout
fig.update_layout(
    title='Time Series of Network Metrics',
    xaxis_title='Timestamp',
    yaxis_title='Value',
    legend_title='KPIs and Parameters',
    hovermode='x unified'
)

# Add range slider
fig.update_layout(
    xaxis=dict(
        rangeslider=dict(visible=True),
        type="date"
    )
)

fig.show()

# %%
def make_scatter_for_prb(df, prb_value):
    df_filtered = df[df['min_prb_ratio'] == prb_value]
    return go.Scatter(
        x=df_filtered['CQI'],
        y=df_filtered['DRB.UEThpDl'] / 1000.0,  # Convert to Mbps
        mode='markers',
        name=f'min_prb_ratio = {prb_value}',
        marker=dict(
            size=8,
            opacity=0.7,
        ),
        hovertemplate='CQI: %{x}<br>Throughput: %{y:.2f} Mbps<extra></extra>'
    )

# Get unique min_prb_ratio values
unique_prb_values = sorted(df['min_prb_ratio'].unique())

# We don't need plots for every min_prb_ratio value, so let's just take every fifth value
unique_prb_values = unique_prb_values[::5]

# Create subplot grid with one subplot per min_prb_ratio value
fig = make_subplots(
    rows=1, 
    cols=len(unique_prb_values),
    subplot_titles=[f'min_prb_ratio = {val}' for val in unique_prb_values],
    shared_yaxes=True
)

# Add a scatter trace for each min_prb_ratio value
for i, prb_value in enumerate(unique_prb_values):
    fig.add_trace(
        make_scatter_for_prb(df, prb_value),
        row=1, 
        col=i+1
    )

# Update layout
fig.update_layout(
    title='Throughput vs. CQI by min_prb_ratio',
    height=500,
    width=200 * len(unique_prb_values),
    showlegend=False
)

# Update axes labels
for i in range(len(unique_prb_values)):
    fig.update_xaxes(title_text="CQI", row=1, col=i+1)
    if i == 0:  # Only add y-axis title to the first subplot
        fig.update_yaxes(title_text="Throughput (Mbps)", row=1, col=i+1)

fig.show()

# %% [markdown]
"""
You generated these time series and scatter plots for your *cleaned* data as
part of the HW for the last session. The plots you generate in your Colab
notebook should match those.

If the plots look as you expect, you can proceed to the next step. If not, maybe
you are not loading your processed dataset.

Once you have verified the dataset, you can proceed to the next step.
"""

# %% [markdown]
"""
## Creating and Training a Simple Linear Regression Model

Now we will create a simple linear regression model to predict the minimum PRB
ratio (`min_prb_ratio`) required to a achieve a various downlink throughput
requirements (`DRB.UEThpDl`) for the priority slice (and emergency responder
device), under a variety of channel conditions (`CQI`).

### Linear Regression Model Class
"""

# %%
# Define the model class which inherits from torch.nn.Module
class LinearRegressionModel(torch.nn.Module):
    """
    A simple linear regression model with batch normalization for predicting the
    minimum PRB ratio required to achieve a given downlink throughput for the
    priority slice under various channel conditions.

    We include normalization and denormalization steps within the model, but
    these steps could be taken as pre/post-processing steps outside of the
    model. For this exercise, we include them within the model for simplicity of
    deployment later.
    """
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(2, 1)  # two input features, one output feature
        
        """
        Apply batch normalization to input features. Normalizing input features
        to have zero mean and unit variance is a common practice that can help
        improve model training. In general, normalizing input features can help
        the model converge faster and reduce the risk of getting stuck in local
        minima.
        
        Here, we are applying batch normalization to the input features before
        passing them through the linear layer. This also allows us to feed the
        features directly to the model without extra preprocessing.
        """
        self.batch_norm = torch.nn.BatchNorm1d(2)
        
        """
        We also need to keep track of the mean and standard deviation of the
        target variable (min_prb_ratio) during training. This is because we
        normalize the target variable during training and need to apply the same
        normalization during inference to get the correct predictions.
        
        For our application, we want to return the predictions in the original
        scale of the target variable (min_prb_ratio). To do this, we need to
        store the mean and standard deviation of the target variable during
        training so we can use them to "denormalize" the predictions after
        inference and return values in the original scale to the xApp that will
        be using the model.
        """
        self.register_buffer('y_mean', torch.zeros(1))
        self.register_buffer('y_std', torch.ones(1))

    def forward(self, x):
        """
        The forward method defines the computation performed at every call of
        the model. In this case, calculating the output from normalized input
        features.
        """
        x_normalized = self.batch_norm(x)
        output = self.linear(x_normalized)
        
        """
        After training, we want to return the predictions in the original scale
        of the target variable (min_prb_ratio). To do this, we need to
        "denormalize" the predictions by applying the mean and standard
        deviation of the target variable that we stored during training.
        """
        if not self.training:
            with torch.no_grad():
                output = output * self.y_std + self.y_mean
                
        return output


# %% [markdown]
"""
### Data Preparation (features, targets, and tensors)

Now that we have defined the model, we need to prepare the data for training. We
will use the `CQI` and `DRB.UEThpDl` columns as input features and the
`min_prb_ratio` column as the target variable. So, we need to extract these
columns from the dataset and create tensors from them.

But what is a tensor? Tensors are multi-dimensional arrays used to represent
data in PyTorch. They are similar to NumPy arrays, but with additional features
that make them suitable for deep learning tasks. E.g.:

1. They can be used on both CPUs and GPUs.
2. They support automatic differentiation (autograd) for gradient-based
   optimization. I.e., they efficiently compute and track the gradients of a
   loss function with respect to the model parameters across all layers of the
   network. This is essential for training deep learning models using
   backpropagation.
3. They can be used to define and store neural network weights, biases, and
   activations, as well as other variables that you define for your model.
"""

# %%
# Extract the input features and target variable
X = torch.tensor(df[['CQI', 'DRB.UEThpDl']].values, dtype=torch.float32)
y = torch.tensor(df['min_prb_ratio'].values, dtype=torch.float32)

# Reshape the target variable to have a shape of (n_samples, 1)
y = y.view(-1, 1)
print(f"X shape: {X.shape}, y shape: {y.shape}")

# %% [markdown]
"""
### Training the Model

Now that we have the data in the form of tensors, we can create an instance of
the `LinearRegressionModel` class and train it on the data. We will use the mean
squared error (MSE) loss function and stochastic gradient descent (SGD) as the
optimizer.
"""

# %%
# Get the device: GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %%
model = LinearRegressionModel()
model.y_mean = y.mean(dim=0, keepdim=True)
model.y_std = y.std(dim=0, keepdim=True)
model.to(device)
X.to(device)
y.to(device)

"""
We'll use the mean squared error (MSE) loss function, which calculates the
average squared difference between the predicted and actual values.

[documentation](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)
"""
criterion = torch.nn.MSELoss() # Mean Squared Error

"""
We will use stochastic gradient descent (SGD) as the optimizer. SGD updates the
model parameters based on the gradients of the loss function with respect to the
model parameters. The learning rate determines the step size at each iteration
of the optimization process. A smaller learning rate means smaller updates to
the model parameters, while a larger learning rate means larger updates.

Choosing a learning rate that is too high can cause the model to diverge, while
a learning rate that is too low can cause the model to converge too slowly. In
practice, you may need to experiment with different learning rates to find the
one that works best for your model and dataset.

A common approach is to start with a small learning rate and gradually increase
it until the training process starts to diverge. Then, you can decrease the
learning rate to find a more stable value.

Alternatively, you can use a learning rate scheduler that automatically adjusts
the learning rate during training based on the model's performance. This can
help improve the model's convergence and reduce the risk of overfitting. In this
case, we will just ue a fixed learning rate of 0.05.

[documentation](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
"""
optimizer = torch.optim.SGD(model.parameters(), lr=.05)

"""
We will train the model for `num_epochs` epochs. An epoch is one complete pass
through the entire training dataset. The number of epochs is a hyperparameter
that you can tune to improve the model's performance.

In general, more epochs mean more training, but also means more risk of
overfitting the model to the training data, which can occur when, e.g., the
model learns the noise in the training data.
"""
num_epochs = 200

"""
Let's create an array to store the loss values for each epoch. This will
allow us to visualize the convergence of the loss function during training.
"""
losses = np.zeros(num_epochs)

for epoch in range(num_epochs):
    model.train()

    """
    Compute predictions and loss for the current epoch. The loss is calculated
    against the normalized target variable (min_prb_ratio).
    """
    y_predicted = model(X)
    loss = criterion(y_predicted, (y - model.y_mean) / model.y_std)

    # Store the loss value for this epoch
    losses[epoch] = loss.item()

    
    """
    The loss.backward() method computes the gradients of the loss function with
    respect to the model parameters. This is done using backpropagation, which
    is an efficient algorithm for computing gradients in neural networks. These
    gradients will be used by the optimizer to update the model parameters.
    """
    loss.backward()
    
    """
    The optimizer.step() method updates the model parameters based on the
    gradients computed during the backward pass. The optimizer.zero_grad()
    method clears the gradients of all optimized tensors. This is important
    because, by default, gradients accumulate in PyTorch. If we don't clear the
    gradients, they will accumulate over multiple iterations, which can lead to
    incorrect updates to the model parameters. (In some cases, accumulating
    gradients can be useful, but with our relatively simple model, we don't need
    to do that.)
    """
    optimizer.step()
    optimizer.zero_grad()

    if (epoch) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Normalized Loss: {loss.item():.4f}')

# %%
# plot the convergence of the loss function
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(num_epochs), y=losses, mode='lines', name='Loss'))
fig.update_layout(title='Normalized Loss Convergence During Training', xaxis_title='Epoch', yaxis_title='Normalized Loss')
fig.show()


# %% [markdown]
"""
### Learned Parameters

Now that the model is trained, let's look at the learned parameters. We can
extract the learned weights and bias from the model and print them. These
parameters define the hyperplane that the model has learned to fit the data. (We
use the term "hyperplane" here because it generalizes to higher dimensional
feature spaces.)
"""

# %%
# Get the learned parameters
learned_weights = model.linear.weight.data.cpu().numpy()
learned_bias = model.linear.bias.data.cpu().numpy()

# Print the learned hyperplane equation
print("\nLearned Hyperplane:")
print(f"min_prb_ratio = {learned_weights[0][0]:.2f}*CQI + {learned_weights[0][1]:.2f}*DRB.UEThpDl + {learned_bias[0]:.2f}")


# %% [markdown]
"""
### Visual Validation

Now let's do some simple visual validation of the model, using predictions for
the required minimum slice resources (`min_prb_ratio`) required for various
channel conditions (`CQI`) and downling throughput requirements (`DRB.UEThpDl`).

We are essentially fitting a plane to the data points in the 3D space defined by
the `CQI`, `DRB.UEThpDl`, and `min_prb_ratio` variables. Let's visualize the
hyperplane fit and discuss for few minutes.
"""

# %%
model.eval()

# Get the learned parameters
learned_weights = model.linear.weight.data.cpu().numpy()
learned_bias = model.linear.bias.data.cpu().numpy()

# Print the learned hyperplane equation (in normalized space)
print("\nLearned Hyperplane (in normalized space):")
print(f"y_normalized = {learned_weights[0][0]:.2f}*x1_normalized + {learned_weights[0][1]:.2f}*x2_normalized + {learned_bias[0]:.2f}")

# Get feature normalization parameters from the batch normalization layer
x_mean = model.batch_norm.running_mean.cpu().numpy().reshape(1, -1)
x_std = torch.sqrt(model.batch_norm.running_var).cpu().numpy().reshape(1, -1)

# Create normalized versions of features and targets using the model's normalization parameters
features_normalized = (X.cpu().numpy() - x_mean) / x_std
targets_normalized = (y.cpu().numpy() - model.y_mean.cpu().numpy()) / model.y_std.cpu().numpy()

# Get original denormalized data
features_denormalized = X.cpu().numpy()
targets_denormalized = y.cpu().numpy()

# Sample every 5th point for clarity in the scatter plot
sample_indices = np.arange(0, len(features_denormalized), 5)
features_sampled = features_denormalized[sample_indices]
targets_sampled = targets_denormalized[sample_indices]

# Create a meshgrid for the hyperplane using denormalized feature ranges
x1_range = np.linspace(features_denormalized[:,0].min(), features_denormalized[:,0].max(), 20)
x2_range = np.linspace(features_denormalized[:,1].min(), features_denormalized[:,1].max(), 20)
X1, X2 = np.meshgrid(x1_range, x2_range)

# Convert meshgrid to normalized space for prediction using the batch normalization parameters
X1_normalized = (X1 - x_mean[0, 0]) / x_std[0, 0]
X2_normalized = (X2 - x_mean[0, 1]) / x_std[0, 1]

# Calculate predictions in normalized space
Y_predicted_normalized = learned_weights[0][0] * X1_normalized + learned_weights[0][1] * X2_normalized + learned_bias[0]

# Convert predictions back to denormalized space
Y_predicted_denormalized = Y_predicted_normalized * model.y_std.cpu().numpy()[0, 0] + model.y_mean.cpu().numpy()[0, 0]

# Create Plotly figure
fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])

# Add scatter plot for data points
scatter = go.Scatter3d(
    x=features_sampled[:,0],
    y=features_sampled[:,1],
    z=targets_sampled.flatten(),
    mode='markers',
    marker=dict(
        size=2,
        color='red',
        opacity=0.8
    ),
    name='Data Points',
    hovertemplate='CQI: %{x:.2f}<br>Throughput: %{y:.2f} Mbps<br>min_prb_ratio: %{z:.2f}<extra></extra>'
)
fig.add_trace(scatter)

# Add surface plot for the hyperplane
surface = go.Surface(
    x=X1, 
    y=X2, 
    z=Y_predicted_denormalized,
    colorscale='Blues',
    opacity=0.7,
    showscale=False,
    name='Predicted Hyperplane',
    hovertemplate='CQI: %{x:.2f}<br>Throughput: %{y:.2f} Mbps<br>Predicted min_prb_ratio: %{z:.2f}<extra></extra>'
)
fig.add_trace(surface)

# Update layout with labels and title
fig.update_layout(
    title='Linear Regression Hyperplane Fit (Denormalized Values)',
    scene=dict(
        xaxis_title='CQI',
        yaxis_title='DRB.UEThpDL (Mbps)',
        zaxis_title='min_prb_ratio',
        aspectmode='auto'
    ),
    legend=dict(
        y=0.99,
        x=0.01,
        font=dict(size=12)
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    width=800,
    height=600
)

fig.show()

# %% [markdown]
"""
## Train/Test Splits Using the Datasets Package

To better evaluate our model's performance, we can split our data into training
and testing sets. This allows us to train on one subset and evaluate on another
subset that the model hasn't seen, giving us a better idea of how well the model
generalizes to new data. (The goal is not to memorize the training data, but to
learn the underlying patterns that can be applied to new data.)

We'll use the datasets package to create this split in a random but reproducible
way.
"""

# %% [markdown]
"""
### Create the Split
"""
# %%
# Create a dataset from our DataFrame
from datasets import Dataset

# Convert the pandas DataFrame to a datasets.Dataset object
full_dataset = Dataset.from_pandas(df)

"""
Split the dataset into training and testing sets (80% train, 20% test). The seed
value is used to ensure that the split is reproducible. This means that every
time you run the code with the same seed value, you will get the same split of
the dataset.
"""
split_dataset = full_dataset.train_test_split(test_size=0.2, seed=42)

# Print information about the split
print(f"Training set size: {len(split_dataset['train'])}")
print(f"Test set size: {len(split_dataset['test'])}")

# Convert back to pandas DataFrames for easier handling
train_df = split_dataset['train'].to_pandas()
test_df = split_dataset['test'].to_pandas()

# %% [markdown]
"""
### Extract Features and Target from Train and Test Sets
"""

# %%
# Extract features and target from train and test sets
X_train = torch.tensor(train_df[['CQI', 'DRB.UEThpDl']].values, dtype=torch.float32)
y_train = torch.tensor(train_df['min_prb_ratio'].values, dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(test_df[['CQI', 'DRB.UEThpDl']].values, dtype=torch.float32)
y_test = torch.tensor(test_df['min_prb_ratio'].values, dtype=torch.float32).view(-1, 1)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# %% [markdown]
"""
### Train on the Test Set

Now we'll train a new model using only the training data and evaluate it on the
test data, which should help ensure that our model generalizes well to unseen
data. We'll use the same model architecture as before, but we'll calculate the
normalization parameters (mean and standard deviation) from the training data
only. This is important because we want to avoid using any information from the
test set during training.
"""

# %%
# Initialize a new model for the train/test experiment
model_split = LinearRegressionModel()

# Calculate normalization parameters from training data only
# Important: we only use training data statistics for normalization
model_split.y_mean = y_train.mean(dim=0, keepdim=True)
model_split.y_std = y_train.std(dim=0, keepdim=True)

# Move everything to the device
model_split.to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model_split.parameters(), lr=0.05)

# Training parameters
num_epochs = 200

# Track training and testing losses
train_losses = np.zeros(num_epochs)
test_losses = np.zeros(num_epochs)

# Train the model
for epoch in range(num_epochs):
    # Training mode
    model_split.train()
    
    # Forward pass with training data
    y_train_pred = model_split(X_train)
    train_loss = criterion(y_train_pred, (y_train - model_split.y_mean) / model_split.y_std)
    train_losses[epoch] = train_loss.item()

    # Backward and optimize
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
    # Evaluation mode - no gradients needed
    model_split.eval()
    with torch.no_grad():
        # Forward pass with test data
        y_test_pred = model_split(X_test)
            # Calculate test loss on normalized data
        test_loss = criterion(
            (y_test_pred - model_split.y_mean) / model_split.y_std, 
            (y_test - model_split.y_mean) / model_split.y_std
        )
        test_losses[epoch] = test_loss.item()
    
    if (epoch) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Normalized Train Loss: {train_loss.item():.4f}, Normalized Test Loss: {test_loss.item():.4f}')

# %% [markdown]
"""
### Plot Both Loss Curves
"""

# %%
# Plot the training and test loss curves
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=np.arange(num_epochs), 
    y=train_losses, 
    mode='lines', 
    name='Training Loss'
))

fig.add_trace(go.Scatter(
    x=np.arange(num_epochs), 
    y=test_losses, 
    mode='lines', 
    name='Test Loss'
))

fig.update_layout(
    title='Loss Convergence with Train/Test Split',
    xaxis_title='Epoch',
    yaxis_title='Loss',
    legend=dict(x=0.02, y=0.98)
)

fig.show()

# %% [markdown]
"""
### Validate Model Performance on Test Set

Now let's evaluate our model's performance on the test set in order to give us a
better indication of how well our model will generalize to unseen data. We'll
calculate some useful performance metrics as well.
"""

# %%
# Evaluate the model on the test set
model_split.eval()

# Make predictions on the test set
with torch.no_grad():
    y_test_pred = model_split(X_test)

# Convert to numpy for easier calculations
y_test_np = y_test.cpu().numpy()
y_test_pred_np = y_test_pred.cpu().numpy()

# Calculate MSE and RMSE
test_mse = np.mean((y_test_np - y_test_pred_np) ** 2)
test_rmse = np.sqrt(test_mse)

"""
Calculate R^2 (coefficient of determination). The R^2 score indicates how well
the model explains the variance in the target variable by comparing the residual
variance (the variance of the errors in the model's predictions) to the total
variance (the variance of the target variable without considering the model at
all).

An R^2 score of 1 indicates that the model perfectly explains the variance in
the target variable, while an R^2 score of 0 indicates that the model does not
explain any of the variance in the target variable. A negative R^2 score
indicates that the model is worse than a simple mean prediction.
"""
y_test_mean = np.mean(y_test_np)
ss_total = np.sum((y_test_np - y_test_mean) ** 2)
ss_residual = np.sum((y_test_np - y_test_pred_np) ** 2)
r_squared = 1 - (ss_residual / ss_total)

print(f"Test MSE: {test_mse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test R^2: {r_squared:.4f}")

# %% [markdown]
"""
## Polynomial Regression Model

Let's see if we can do better with another model implementation. Polynomial
regression is a type of regression analysis in which the relationship between
the independent variables (input features) and the dependent variable (target
variable) is modeled as an nth degree polynomial. This allows for more complex
relationships between the input features and the target variable compared to
linear regression, which assumes a linear relationship.

The polynomial regression model can be implemented by transforming the input
features into polynomial features. This is done by creating new features that
are powers or combinations of the original features. For example, if we have two
input features `x1` and `x2`, we can create polynomial features like `x1^2`,
`x2^2`, `x1*x2`, etc. The transformed features are then used in the linear
regression model. So, technically, we are still using linear regression, but the
input features are transformed to allow for more complex relationships.

In this case, we will use a polynomial regression model with degree 2. This
means we will create polynomial features up to the second degree. The model will
be able to learn quadratic relationships between the input features and the
target variable. This can help improve the model's performance if the
relationship between the input features and the target variable is not linear.
"""

%% [markdown]
"""
### Polynomial Regression Model Class
"""
# %%
# Define the polynomial regression model
class PolynomialRegressionModel(torch.nn.Module):
    def __init__(self, degree=2):
        super(PolynomialRegressionModel, self).__init__()
        self.degree = degree
        
        """
        Calculate number of polynomial features for 2 input features with degree
        n. For 2 features with degree 2: x1, x2, x1^2, x1*x2, x2^2 = 5 features.
        We subtract 1 because we start from degree 1 rather than 0.
        """
        n_poly_features = int((degree + 1) * (degree + 2) / 2) - 1
        
        # Apply batch normalization to expanded polynomial features
        self.batch_norm = torch.nn.BatchNorm1d(n_poly_features)
        
        # Linear layer now accepts polynomial features as input
        self.linear = torch.nn.Linear(n_poly_features, 1)
        
        # Register buffers to store the mean and standard deviation of the output features
        self.register_buffer('y_mean', torch.zeros(1))
        self.register_buffer('y_std', torch.ones(1))

    def _polynomial_features(self, x):
        """
        Generate polynomial features up to the specified degree.
        For input [x1, x2], with degree=2, this generates [x1, x2, x1^2, x1*x2, x2^2]
        """
        batch_size = x.shape[0]
        x1 = x[:, 0].view(-1, 1)
        x2 = x[:, 1].view(-1, 1)
        
        # Start with degree 1 terms (original features)
        poly_features = [x1, x2]
        
        # Add higher degree terms
        for d in range(2, self.degree + 1):
            for i in range(d + 1):
                # Add term x1^(d-i) * x2^i
                term = torch.pow(x1, d-i) * torch.pow(x2, i)
                poly_features.append(term)
        
        """
        Concatenate all polynomial features into a single tensor. The resulting
        tensor will have shape (batch_size, n_poly_features), where
        n_poly_features is the number of polynomial features generated. For
        example, if we have 2 input features and degree=2, the resulting tensor
        will have shape (batch_size, 5).
        """
        return torch.cat(poly_features, dim=1)

    def forward(self, x):
        # First transform input to polynomial features
        x_poly = self._polynomial_features(x)
        
        """
        We need to normalize the polynomial features before passing them to the linear
        layer.
        """
        x_poly_normalized = self.batch_norm(x_poly)
        
        # Apply linear transformation to normalized polynomial features
        output = self.linear(x_poly_normalized)
        
        # Denormalize output during inference
        if not self.training:
            with torch.no_grad():
                output = output * self.y_std + self.y_mean
                
        return output

# %% [markdown]
"""
### Training Polynomial Regression Model

Now let's train and evaluate the polynomial regression model with degree=2 that
we defined above. We'll use the same methodology as we did for the linear
regression model and then compare the performance.
"""

# %%
# Initialize the polynomial regression model with degree=2
poly_model = PolynomialRegressionModel(degree=2)

# Calculate normalization parameters from training data only (same as linear model)
poly_model.y_mean = y_train.mean(dim=0, keepdim=True)
poly_model.y_std = y_train.std(dim=0, keepdim=True)

# Move model to device
poly_model.to(device)

# Define loss function and optimizer (same as linear model)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(poly_model.parameters(), lr=0.05)

# Training parameters
num_epochs = 200

# Track training and testing losses
poly_train_losses = np.zeros(num_epochs)
poly_test_losses = np.zeros(num_epochs)

# Train the polynomial model
for epoch in range(num_epochs):
    # Training mode
    poly_model.train()
    
    # Forward pass with training data
    y_train_pred = poly_model(X_train)
    train_loss = criterion(y_train_pred, (y_train - poly_model.y_mean) / poly_model.y_std)
    poly_train_losses[epoch] = train_loss.item()

    # Backward and optimize
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
    # Evaluation mode - no gradients needed
    poly_model.eval()
    with torch.no_grad():
        # Forward pass with test data
        y_test_pred = poly_model(X_test)
        # Calculate test loss on normalized data
        test_loss = criterion(
            (y_test_pred - poly_model.y_mean) / poly_model.y_std, 
            (y_test - poly_model.y_mean) / poly_model.y_std
        )
        poly_test_losses[epoch] = test_loss.item()
    
    if (epoch) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Poly Train Loss: {train_loss.item():.4f}, Poly Test Loss: {test_loss.item():.4f}')

# %% [markdown]
"""
### Comparing Training Loss Curves for Linear vs. Polynomial Models

Let's compare the training and test loss curves for both models to see how they
converge.
"""

# %%
# Plot loss curves for both models on the same plot
fig = make_subplots(rows=1, cols=2, subplot_titles=('Training Loss', 'Test Loss'))

# Training loss curves
fig.add_trace(
    go.Scatter(
        x=np.arange(num_epochs), 
        y=train_losses, 
        mode='lines', 
        name='Linear Model (Train)',
        line=dict(color='blue')
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=np.arange(num_epochs), 
        y=poly_train_losses, 
        mode='lines', 
        name='Polynomial Model (Train)',
        line=dict(color='red')
    ),
    row=1, col=1
)

# Test loss curves
fig.add_trace(
    go.Scatter(
        x=np.arange(num_epochs), 
        y=test_losses, 
        mode='lines', 
        name='Linear Model (Test)',
        line=dict(color='blue', dash='dash')
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(
        x=np.arange(num_epochs), 
        y=poly_test_losses, 
        mode='lines', 
        name='Polynomial Model (Test)',
        line=dict(color='red', dash='dash')
    ),
    row=1, col=2
)

fig.update_layout(
    title='Normalized Loss Convergence: Linear vs. Polynomial Models',
    xaxis_title='Epoch',
    yaxis_title='Loss',
    height=400,
    width=1000
)

fig.show()

# %% [markdown]
"""
#### Hmm... do you notice anything interesting about the loss curves above? Anything we might try to improve the training?
"""

# %% [markdown]
"""
### Evaluating the Polynomial Model

Now that we've tuned things a bit, let's evaluate the polynomial model's
performance on the test set, similar to how we evaluated the linear model.
"""

# %%
# Evaluate the polynomial model on the test set
poly_model.eval()

# Make predictions on the test set
with torch.no_grad():
    y_test_pred_poly = poly_model(X_test)

# Convert to numpy for easier calculations
y_test_np = y_test.cpu().numpy()
y_test_pred_poly_np = y_test_pred_poly.cpu().numpy()

# Calculate MSE and RMSE for polynomial model
poly_test_mse = np.mean((y_test_np - y_test_pred_poly_np) ** 2)
poly_test_rmse = np.sqrt(poly_test_mse)

# Calculate R^2 for polynomial model
y_test_mean = np.mean(y_test_np)
ss_total = np.sum((y_test_np - y_test_mean) ** 2)
ss_residual_poly = np.sum((y_test_np - y_test_pred_poly_np) ** 2)
poly_r_squared = 1 - (ss_residual_poly / ss_total)

print(f"Polynomial Model - Test MSE: {poly_test_mse:.4f}")
print(f"Polynomial Model - Test RMSE: {poly_test_rmse:.4f}")
print(f"Polynomial Model - Test R^2: {poly_r_squared:.4f}")

# Get linear model predictions for comparison
model_split.eval()
with torch.no_grad():
    y_test_pred_linear = model_split(X_test)

y_test_pred_linear_np = y_test_pred_linear.cpu().numpy()

# %% [markdown]
"""
## Comparing the Two Models

Let's compare the performance of the linear and polynomial regression models in
terms of the metrics we've been recording.
"""

# %% [markdown]
"""
### Model Performance Metrics
"""

# %%
# Create a comparison table of metrics
metrics_comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Polynomial Regression (degree=2)'],
    'MSE': [test_mse, poly_test_mse],
    'RMSE': [test_rmse, poly_test_rmse],
    'R^2': [r_squared, poly_r_squared]
})

print("Model Performance Comparison:")
print(metrics_comparison)

# %% [markdown]
"""
### Visualizing the Model Fits

Let's also create a side-by-side 3D visualization of both models to compare
their fits.
"""

# %%
# Get linear model predictions for the same grid
model_split.eval()
with torch.no_grad():
    grid_preds_linear = model_split(grid_tensor).cpu().numpy()

# Reshape predictions to match grid
Z_linear = grid_preds_linear.reshape(X1.shape)

# Create figure with two subplots
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scene'}, {'type': 'scene'}]],
    subplot_titles=('Linear Model', 'Polynomial Model (Degree=2)')
)

# Linear model surface
fig.add_trace(
    go.Surface(
        x=X1, 
        y=X2, 
        z=Z_linear,
        colorscale='Blues',
        opacity=0.7,
        name='Linear Model',
        showscale=False
    ),
    row=1, col=1
)

# Test data points for linear model subplot
fig.add_trace(
    go.Scatter3d(
        x=X_test_sample[:,0],
        y=X_test_sample[:,1],
        z=y_test_sample.flatten(),
        mode='markers',
        marker=dict(
            size=3,
            color='black',
            opacity=0.7
        ),
        name='Test Data',
        showlegend=False
    ),
    row=1, col=1
)

# Polynomial model surface
fig.add_trace(
    go.Surface(
        x=X1, 
        y=X2, 
        z=Z_poly,
        colorscale='Reds',
        opacity=0.7,
        name='Polynomial Model',
        showscale=False
    ),
    row=1, col=2
)

# Test data points for polynomial model subplot
fig.add_trace(
    go.Scatter3d(
        x=X_test_sample[:,0],
        y=X_test_sample[:,1],
        z=y_test_sample.flatten(),
        mode='markers',
        marker=dict(
            size=3,
            color='black',
            opacity=0.7
        ),
        name='Test Data'
    ),
    row=1, col=2
)

# Update subplot scene properties
for i in [1, 2]:
    fig.update_scenes(
        xaxis_title='CQI',
        yaxis_title='DRB.UEThpDl',
        zaxis_title='min_prb_ratio',
        aspectmode='auto',
        row=1, col=i
    )

# Update layout
fig.update_layout(
    title='Model Comparison: Linear vs. Polynomial Surface Fits',
    width=1200,
    height=600,
    margin=dict(l=0, r=0, b=0, t=50)
)

fig.show()

# %% [markdown]
"""
## Exporting the Model to Hugging Face and Verifying We can Load it Back

Let's pretend that our work is done here and save a trained model to HuggingFace
so we can use it later. We will first export the model to ONNX format.

ONNX (Open Neural Network Exchange) is an open format for representing machine
learning models. It allows models to be transferred between different frameworks
and platforms, making it easier to deploy models in various environments. ONNX
provides a standardized way to represent models, enabling interoperability
between different deep learning frameworks such as PyTorch, TensorFlow, and
others.

[ONNX documentation](https://onnx.ai/onnx/intro/)

We're using ONNX to export our trained PyTorch model so our inference server can
simply use the ONNX runtime to load the model and run inference on it without
any additional dependencies or libraries, and without needing to worry about
importing the classes and methods you designed for your models.

We'll upload the polynomial regression model as an example.
"""

# %% [markdown]
"""
### Save the Polynomial Regression Model to Hugging Face
"""

# %%
# Save the polynomial regression model to ONNX and upload to Hugging Face
poly_model_name = "polynomial_regression_model"
poly_model_version = "0.0.1"  # alpha version

"""
We'll save some metadata about the model that may be useful.
"""
poly_metadata_props = {
    "version": poly_model_version,
    "training_date": datetime.datetime.now().isoformat(),
    "framework": f"PyTorch {torch.__version__}",
    "dataset": f"{dataset_name}",
    "metrics": json.dumps({
        "mse": f"{poly_test_mse}",
        "rmse": f"{poly_test_rmse}",
        "r2": f"{poly_r_squared}"
    }),
    "description": f"Polynomial regression model for min PRB prediction based on CQI and DRB.UEThpDl.",
    "input_features": json.dumps(["CQI", "DRB.UEThpDl"]),
    "output_features": json.dumps(["min_prb_ratio"]),
    "polynomial_degree": poly_model.degree,
    "model_type": "polynomial_regression"
}

# Create temp directory
poly_temp_dir = tempfile.mkdtemp()
poly_model_path = os.path.join(poly_temp_dir, f"{poly_model_name}_v{poly_model_version}.onnx")

# Export the model to ONNX
dummy_input = torch.randn(1, 2)  # Example input
torch.onnx.export(
    poly_model, 
    dummy_input, 
    poly_model_path, 
    verbose=True, 
    input_names=["input"], 
    output_names=["output"], 
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

# Create repository for polynomial model
poly_model_repo = f"cyberpowder/{poly_model_name}_v{poly_model_version}"

# Create metadata JSON file
poly_metadata_path = os.path.join(poly_temp_dir, f"{poly_model_name}_v{poly_model_version}_metadata.json")
with open(poly_metadata_path, 'w') as f:
    json.dump(poly_metadata_props, f, indent=2)

# Create or ensure repository exists
try:
    hf_api.create_repo(poly_model_repo, private=True, token=token)
    print(f"Created new repository: {poly_model_repo}")
except Exception as e:
    print(f"Repository may already exist or error creating it: {e}")

# Upload ONNX model to Hugging Face
print(f"Uploading polynomial ONNX model to Hugging Face: {poly_model_repo}")
hf_api.upload_file(
    path_or_fileobj=poly_model_path,
    repo_id=poly_model_repo,
    path_in_repo=f"{poly_model_name}_v{poly_model_version}.onnx",
    token=token
)

# Upload metadata to Hugging Face
print(f"Uploading polynomial model metadata to Hugging Face: {poly_model_repo}")
hf_api.upload_file(
    path_or_fileobj=poly_metadata_path,
    repo_id=poly_model_repo,
    path_in_repo=f"{poly_model_name}_v{poly_model_version}_metadata.json",
    token=token
)

print(f"Polynomial model and metadata successfully uploaded to Hugging Face: {poly_model_repo}")

# %% [markdown]
"""
### Verify ONNX Model Download and Inference

Now let's make sure that we can load the model back from HuggingFace and run
inference using the ONNX runtime.
"""

# %%
# List files in the repo to confirm upload was successful
print(f"Files in repository {poly_model_repo}:")
model_files = hf_api.list_repo_files(poly_model_repo, token=token)
for file in model_files:
    print(f"  - {file}")

# Create temporary directory for downloaded files
poly_download_dir = tempfile.mkdtemp()

try:
    # Download model from Hugging Face
    print(f"\nDownloading polynomial model from Hugging Face...")
    poly_model_path = hf_hub_download(
        repo_id=poly_model_repo,
        filename=f"{poly_model_name}_v{poly_model_version}.onnx",
        token=token,
        local_dir=poly_download_dir
    )
    
    # Download metadata
    print(f"Downloading polynomial model metadata from Hugging Face...")
    poly_metadata_path = hf_hub_download(
        repo_id=poly_model_repo,
        filename=f"{poly_model_name}_v{poly_model_version}_metadata.json",
        token=token,
        local_dir=poly_download_dir
    )
    
    # Load metadata
    with open(poly_metadata_path, 'r') as f:
        poly_metadata = json.load(f)
    
    print(f"\nPolynomial model metadata:")
    print(f"  Version: {poly_metadata.get('version')}")
    print(f"  Polynomial degree: {poly_metadata.get('polynomial_degree')}")
    print(f"  Description: {poly_metadata.get('description')}")
    print(f"  Framework: {poly_metadata.get('framework')}")
    print(f"  Metrics: {poly_metadata.get('metrics')}")
    
    # Load model with ONNX Runtime
    print(f"\nLoading polynomial model for inference...")
    poly_session = ort.InferenceSession(poly_model_path)
    
    # Some sample data for inference
    # Multiplying the DRB.UEThpDL here by 1000 to convert Mbps to Kbps (model expects Kbps)
    sample_inputs = [
        [6.0, 1.0 * 1000.0],    # Low CQI, low throughput
        [10.0, 100.0 * 1000.0],  # Medium CQI, medium throughput
        [15.0, 300.0 * 1000.0]   # High CQI, high throughput
    ]
    
    input_tensor = np.array(sample_inputs, dtype=np.float32)
    
    # Run inference with polynomial model
    print(f"\nRunning inference with sample data...")
    poly_outputs = poly_session.run(None, {"input": input_tensor})
    
    # Print results as a table
    print("\nPrediction Results:")
    print("------------------------------------------------------")
    print("   CQI   | Throughput (Mbps) | Predicted min_prb_ratio")
    print("------------------------------------------------------")
    for i, sample in enumerate(sample_inputs):
        print(f"  {sample[0]:5.1f}  |      {sample[1]/1000.0:7.1f}     |        {poly_outputs[0][i][0]:7.2f}")
    print("------------------------------------------------------")
except Exception as e:
    print(f"Error during download or inference: {e}")

# %% [markdown]
"""
### Discussion

Notice any issues with the predictions? There is a significant issue we've
ignored up until now and I chose the sample inputs to illustrate it. Recall
that, for our application, the priority slice will never have less than 50% of
the resources...

Also, the CQI feature and the min_prb_ratio target have a certain characteristic
that doesn't mesh well with a regression model...

Maybe have a look at the dataframe from your dataset again.
"""

# %% [markdown]
"""
### Examine the Dataset Again
"""
# %%
# Show the first few rows of the DataFrame
df.head()

# %% [markdown]
"""
After some discussion, we will add more detail to the IH9 O-RAN + ML assignment
on canvas, and you can start working on it for the remainder of today's session,
and continue as HW.
"""