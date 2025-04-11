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
# O-RAN Dataset Analysis and Processing

1. Brief HuggingFace intro
  1. Logging in to HuggingFace
  2. Creating and uploading datasets to HuggingFace
  3. Downloading datasets from HuggingFace
2. Analyzing the O-RAN slicing dataset
  1. Loading the dataset
  2. Some initial data processing
  3. Visualizing the dataset
  4. Brainstorm and apply further data processing
  5. Uploading the processed dataset to HuggingFace

Prerequisites:

- Read Section VI (AI/ML Workflows) of the [NEU ORAN paper](https://utah.instructure.com/courses/1045795/files/170447527?wrap=1)
- Join the [HuggingFace CyberPowder organization](https://huggingface.co/cyberpowder)
  - Instructions [here](https://utah.instructure.com/courses/1045795/assignments/15915757)

## Note: Don't just run the whole notebook. There are some cells that will require interaction. Read the comments and understand what each cell does. 
"""

# %%
# Install required packages (various other required packages are already available in the colab environment)
# !pip install datasets

# %%
# Import required packages
import datetime

import datasets
import huggingface_hub as hf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots


# %% [markdown]
"""
## 1. Brief HuggingFace intro
"""

# %% [markdown]
"""
### Logging in to HuggingFace

Using the HuggingFace API requires and access token. We will walk through this
now, but there are some instructions
[here](https://app.excalidraw.com/s/8g7kivZ39v0/8x63aN6Ps5B?element=Z4ECPgkcwhcUUXJmXPnyA)
as well.

Running the cell below will prompt you to log in to HuggingFace using the
generated access token, so let's go generate one.
"""

# %%
hf.notebook_login()

# %% [markdown]
"""
Check that you're logged in (ignore the warning about adding the token as a
Colab secret).
"""

# %%
username = hf.whoami()['name']
print(f"Logged in as {username}")

# %% [markdown]
"""
From here on out, calls to the HuggingFace API should be automatically
authenticated with your access token.
"""

# %% [markdown]
"""
### Creating and Uploading Datasets to Hugging Face

Let's create some random data to use as a dataset. We'll use numpy to generate
random features and targets and throw them into a pandas DataFrame
"""

# %%
random_features = np.random.rand(100, 2)  # 100 samples, 2 features
random_targets = np.random.rand(100, 1)  # 100 samples, 1 target
df = pd.DataFrame(random_features, columns=["feature1", "feature2"])
df["target"] = random_targets
df

# %% [markdown]
"""
Now we'll use the datasets library to create a dataset from the pandas DataFrame.
You could also create a dataset from a csv file, json file, etc.
"""

# %%
dataset = datasets.Dataset.from_pandas(df)
dataset

# %% [markdown]
"""
Now let's create a dataset repository on HuggingFace to store the dataset we just created.
We'll use the current date and time to make the dataset name unique.  We'll also
make the dataset private so that only you can access it.
"""

# %%
dummy_repo_name = f"{username}/dummy-datasets"
dummy_dataset_name = f"dummy-dataset-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

# (this will raise an error if the repository already exists, but we'll ignore that for now)
hf_api = hf.HfApi()
try:
    hf_api.create_repo(dummy_repo_name)
except Exception as e:
    print(f"Error creating repository: {e}")

# %% [markdown]
"""
Now we'll push the dataset to the repository we just created.  If the dataset
already exists, and there are no changes, HuggingFace will not create a new
version/commit. We'll make the dataset private so that only you can access it.
"""

# %%
dataset.push_to_hub(dummy_repo_name, private=True)

# %% [markdown]
"""
### Downloading the dataset

Now let's make sure we can download the dataset we just uploaded.  We'll first
use the HuggingFace API to list the datasets in the repository. You should see
the dataset we just uploaded in the list of datasets. It may be the only dataset
in the list if you haven't uploaded any others.
"""

# %%
my_datasets = hf_api.list_datasets(author=username)
for ds in my_datasets:
    print(f"Dataset: {ds.id}")

# %% [markdown]
"""
Let's download the dataset we just uploaded using the datasets package.
"""

# %%
dataset = datasets.load_dataset(dummy_repo_name)
dataset

# %% [markdown]
"""
We can now access the dataset as a dictionary. Since we didn't specify a
train/test split when we uploaded the dataset, all of the data is in the "train"
key in the dataset dictionary.
"""

# %%
df = dataset['train'].to_pandas()
df

# %% [markdown]
"""
## 2. Analyzing the O-RAN slicing dataset

Now, let's move on to interacting with the O-RAN slicing dataset that we'll be
processing today for use in next Friday's session, where you will each create,
train, and validate a model using PyTorch.

This dataset consists of network metrics collected during a long run of the
example experiment we looked at earlier. The data were pulled from the data lake
and pushed to HuggingFace. You can find the dataset at:
https://huggingface.co/datasets/cyberpowder/cyberpowder-network-metrics

"""


# %% [markdown]
"""
### Loading the slicing dataset

We'll use the datasets library to again load the dataset using the appropriate
repo name and the dataset configuration name (default) used when it was uploaded
to HuggingFace.
"""

# %%
cp_repo_name = "cyberpowder/cyberpowder-network-metrics"
oran_slicing_dataset = datasets.load_dataset(cp_repo_name, "default")
oran_slicing_dataset

# %% [markdown]
"""
### Some initial data processing (part of step ii. as described in the paper)

The first thing to be aware of is that we don't need all of the data that came
out of the data lake for this experiment.

First, we'll turn the dataset into a pandas DataFrame so that we can use the
pandas package to process the data. Again, this dataset was created without a
train/test split, so all of the data is in the "train" key in the dataset
dictionary.
"""

# %%
oran_slicing_df = oran_slicing_dataset['train'].to_pandas()
oran_slicing_df.describe()

# %% [markdown]
"""
Note that we have KPIs for two UEs as expected, but our simple application (and
future model) only cares about performance guarantees for the emergency responder
UE, which holds `ue_id` 1 in this dataset. The second UE, which happens to hold
`ue_id` 3 in the dataset (for reasons that aren't important here), is the
consumer UE.

Before we start trying to further understand the data, let's remove the
consumer UE from the dataset. We can do this by filtering the DataFrame to
only include rows for `ue_id` 1.
""" 

# %%
oran_slicing_df_ue1 = oran_slicing_df[oran_slicing_df['ue_id'] == 1]
oran_slicing_df_ue1.head()

# %% [markdown]
"""
It might be useful in the future to have direct access to this filtered dataset,
so let's create a new dataset repo under our personal HuggingFace account, and
upload the filtered dataset to that repo. We'll use the configuration name
"emergency-responder-data" to indicate that this dataset only contains data for the
emergency responder UE.
"""

# %%
# Create a new dataset from the filtered DataFrame
my_dataset_repo_name = f"{username}/cyberpowder-network-metrics"
my_dataset_config_name = "emergency-responder-data"

# (this will raise an error if the repository already exists, but we'll ignore
# that for now, since it will probably only happen to me)
try:
    hf_api.create_repo(my_dataset_repo_name)
except Exception as e:
    print(f"Error creating repository: {e}")

# %%
# Push the filtered dataset to the new repository
oran_slicing_dataset_ue1 = datasets.Dataset.from_pandas(oran_slicing_df_ue1)
oran_slicing_dataset_ue1.push_to_hub(
    my_dataset_repo_name,
    config_name=my_dataset_config_name,
    private=True,
)

# %% [markdown]
"""
### Visualizing the dataset

Now that we have the emergency responder dataset, let's take a closer look at
it. We'll use the plotly package to create some interactive plots.
"""

# %% [markdown]
"""
Let's make a copy of our DataFrame with a short name to make the rest of the
code less verbose. We'll also convert the timestamp column to a datetime object
so that we can use it as the x-axis in our plots.
"""

# %%
df = oran_slicing_df_ue1.copy()
df['timestamp'] = pd.to_datetime(df['timestamp'])

# %% [markdown]
"""
Now let's look at the key KPIs and experimental parameters in the dataset.
"""

# %%
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

# %% [markdown]
"""
Go ahead and play around with the figure a bit. Do you notice anything interesting?

There are several unsurprising things about the data, but some of it looks a bit strange. How? Why?

We'll move on to the next plot after some discussion.
"""

# %% [markdown]
"""
OK. Let's look at another view of the data. This time we'll make separate
scatter plots showing DRB.UEThpDl vs. CQI for different min_prb_ratio values.
"""

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
Now take some time to examine this set of figures. Any new insights?
"""
# %% [markdown]
"""
### Brainstorm and apply further data processing

If you are a data science or ML expert (or even a budding one), how might you
further process the data to make training a model more effective? What kind of
model do you think you might use? 

Remember, the goal for our emergency responder application is to predict the
required min_prb_ratio to meet a given DRB.UEThpDl throughput requirement for a
given CQI value.

If you are used to using other tools for understanding and processing data,
there's a good chance that the Colab environment already includes them. If not,
you can use the !pip command to install them.
"""

# %% [markdown]
"""
Continue with your own data processing. We can discuss as you work.

The HW assignment for this session is to:

1. Process the dataset in ways that will make training a model more effective
2. Upload your processed dataset to HuggingFace when done (upload under your
   user account, not the CyberPowder Org, and keep it private for now)
3. Generate both of the plotly figures we created above using your processed
   dataset and save them using the "Download Plot as PNG" option in the plotly
   figure menu
4. Save the code snippets you used for your data processing
5. Generate a brief report that includes:
  - The code snippets you used to further process the dataset 
  - The plot images you saved 
  - Your reasoning behind the data processing steps you took

You can use these processd datasets in the next CyberPowder session to train a
model. The complete homework description will also be posted on the course
website later.
"""
