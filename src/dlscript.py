import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn
from torch_geometric.utils import to_dense_adj
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# use CUDA for faster computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# path to Aqueduct dataset
gdb_path = "Aq40_Y2023D07M05.gdb"

# reading file with pyogrio
wri_gdf = gpd.read_file(gdb_path, layer="baseline_annual", engine="pyogrio")

features_list = [
    "bws_raw",
    "bwd_raw",
    "iav_raw",
    "sev_raw",
    "gtd_raw",
    "rfr_raw",
    "cfr_raw",
    "drr_raw",
    "ucw_raw",
    "cep_raw",
    "udw_raw",
    "usa_raw",
    "rri_raw",
]

metadata_cols = ["string_id", "geometry"]
all_cols = metadata_cols + features_list

wri_subset = wri_gdf[all_cols].dropna(subset=["string_id"])

for col in features_list:
    wri_subset[col] = pd.to_numeric(wri_subset[col], errors="coerce").fillna(0.0)
    wri_subset.loc[wri_subset[col] < 0, col] = 0.0


sample_basins = wri_subset.head(8192)

raw_matrix = torch.tensor(sample_basins[features_list].values, dtype=torch.float32)

aqueduct_matrix = raw_matrix.view(64, 128, len(features_list)).to(device)

# set random seed
torch.manual_seed(42)

# climate feature tracking for different regions/time line. Features are temperature, humidity, and CO2 levels.

torch.manual_seed(42)
batch_size, seq_len, feature_dim = aqueduct_matrix.shape  # 64, 128, 13
hidden_dim = 4


# LSTM forecasting

lstm_input = aqueduct_matrix

lstm_layer = nn.LSTM(
    input_size=feature_dim, hidden_size=hidden_dim, batch_first=True
).to(device)
lstm_out, _ = lstm_layer(lstm_input)

tomorrow_prediction = lstm_out[:, -1, :]
print("LSTM (Temporal Forecasting):")
print(f"Input Shape: {list(lstm_input.shape)} (Batches, Sequence Length, Indicators)")
print(f"Prediction Vector Shape: {list(tomorrow_prediction.shape)}\n")

print("LSTM Chronological Progression:")
print("LSTM Chronological Progression (First Batch Sample):")
for step in range(
    10
):  # Printing the first 10 steps of the 128-length sequence to avoid scrolling walls of text
    print(
        f"Sequence Step {step+1} Accumulated Momentum Vector: {lstm_out[0, step].detach().cpu().numpy()}"
    )


feature_variance = lstm_out.std(dim=(0, 1)).detach().cpu().numpy()
print(f"Feature Variance: {feature_variance}")
print(f"Index with the Highest Variance (Max: {feature_variance.max():.4f}) ")

# Transformers (long term connection)

transformer_input = lstm_input

transformer_layer = nn.TransformerEncoderLayer(
    d_model=feature_dim, nhead=1, dim_feedforward=hidden_dim, batch_first=True
).to(device)
transformer_out = transformer_layer(transformer_input)

print("Transformer (Long-term Connection):")
print(f"Input Shape: {list(transformer_input.shape)}")
print(f"Output Shape: {list(transformer_out.shape)}\n")


_, attn_weights = transformer_layer.self_attn(
    transformer_input, transformer_input, transformer_input
)
print("Transformer Attention Blueprint:")
print(attn_weights[0].detach().cpu().numpy())

mean_attention = attn_weights.mean(dim=0).detach().cpu().numpy()

most_attended_step = mean_attention.sum(axis=0).argmax()

print(f"Attention Apex Identified at Sequence Index: {most_attended_step}")

# GNN (spread prediction)

gnn_nodes = raw_matrix.to(device)

# Dynamically construct a continuous chain of connections for all 8,192 nodes
sources = torch.arange(0, 8191)
targets = torch.arange(1, 8192)
edge_index = torch.stack(
    [torch.cat([sources, targets]), torch.cat([targets, sources])], dim=0
).to(device)

gcn_layer = geom_nn.GCNConv(in_channels=feature_dim, out_channels=hidden_dim).to(device)
gnn_out = gcn_layer(gnn_nodes, edge_index)

print("GNN (Geospatial Topology):")
print(f"Node Matrix Shape: {list(gnn_nodes.shape)} (Interconnected Watershed Nodes)")
print(
    f"Edge Blueprint Shape: {list(edge_index.shape)} (Physical geographic boundaries)"
)

print(f"Output Shape: {list(gnn_out.shape)}")

spatial_grid = to_dense_adj(edge_index)[0]
print("GNN Grid (First 5x5 sub-matrix sample):")
print(spatial_grid[:5, :5].detach().cpu().numpy())

mean_gnn_output = gnn_out.mean(dim=1).detach().cpu().numpy()
print(
    f"Max Risk Score: {mean_gnn_output.max():.2f} | Min Risk Score: {mean_gnn_output.min():.2f}"
)


# Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    "Data Footprints across Architectural Sub-modules", fontsize=14, fontweight="bold"
)

# LTSM Momemtum states plot
lstm_data = lstm_out[0, :20].detach().cpu().numpy()
axes[0].plot(lstm_data, linewidth=2)
axes[0].set_title("LSTM Sequential State Tracking")
axes[0].set_xlabel("Timeline Progression (Steps)")
axes[0].set_ylabel("Accumulated Vector Value")
axes[0].grid(True, linestyle="--", alpha=0.6)

# Transformer self attention heatmap
attn_matrix_sample = attn_weights[0, :20, :20].detach().cpu().numpy()
im1 = axes[1].imshow(attn_matrix_sample, cmap="viridis", aspect="auto")
axes[1].set_title("Transformer Global Attention Core")
axes[1].set_xlabel("Key Focus Indices")
axes[1].set_ylabel("Query Basin Targets")
fig.colorbar(im1, ax=axes[1], label="Calculated Correlation Weight")

# GNN Grid Plot
gnn_grid_sample = spatial_grid[:20, :20].detach().cpu().numpy()
im2 = axes[2].imshow(gnn_grid_sample, cmap="plasma", aspect="auto")
axes[2].set_title("GNN Grid")
axes[2].set_xlabel("Target Spatial Node")
axes[2].set_ylabel("Source Spatial Node")
fig.colorbar(im2, ax=axes[2], label="Edge Boundary Open (1) / Closed (0)")


plt.tight_layout()
output_filename = "climate_model_differences.png"
plt.savefig(output_filename, dpi=300)
