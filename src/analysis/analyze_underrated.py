import pandas as pd
import numpy as np
import sys
from collections import Counter
from math import log

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

import argparse
import os

# ============================================================
# 0. PATHS  (EDIT IF NEEDED)
# ============================================================
def get_args():
    parser = argparse.ArgumentParser(description="Analyze underrated restaurants")
    parser.add_argument("--input-dir", type=str, default="data/raw", help="Input directory containing raw CSVs")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory for analyzed data")
    parser.add_argument("--city-name", type=str, default="london", help="City name")
    parser.add_argument("--boroughs-file", type=str, default="data/raw/london_boroughs.geojson", help="Path to boroughs geojson")
    return parser.parse_known_args()[0]

args = get_args()

PATH_REST = os.path.join(args.input_dir, f"{args.city_name}_restaurants.csv")
PATH_DET = os.path.join(args.input_dir, f"{args.city_name}_restaurant_details.csv")
PATH_BOROUGHS = args.boroughs_file

# Check if files exist, if not try sample
if not os.path.exists(PATH_DET):
    print(f"Warning: {PATH_DET} not found. Trying sample data...")
    PATH_DET = "data/sample/london_restaurant_details.csv"
    # For basic restaurants, we might not have a sample, so we might need to handle that.
    # But let's assume if details are missing, we might want to use sample details for both if possible or fail gracefully.
    if not os.path.exists(PATH_REST):
         # If basic list is missing, we can try to infer or just use details if the merge isn't strictly required for all columns.
         # However, the code merges on place_id.
         pass 

# ============================================================
# 1. LOAD + RATING MODEL (hype_residual)
# ============================================================
# Handle missing basic file if we are using sample details
if os.path.exists(PATH_REST):
    df_basic = pd.read_csv(PATH_REST)
else:
    # Create dummy basic df if missing, just to allow merge
    df_basic = pd.DataFrame(columns=["place_id", "grid_id"])

df_det = pd.read_csv(PATH_DET)

df = df_det.merge(
    df_basic[["place_id", "grid_id"]],
    on="place_id",
    how="left"
)

# ----- feature engineering -----
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["user_ratings_total"] = pd.to_numeric(df["user_ratings_total"], errors="coerce").fillna(0)
df["log_reviews"] = np.log1p(df["user_ratings_total"])
df["price_level"] = pd.to_numeric(df.get("price_level"), errors="coerce")

# cuisine
df["cuisine"] = (
    df.get("cuisine_detected_ext")
      .fillna(df.get("cuisine_detected"))
      .fillna("unknown")
      .astype(str).str.lower()
)

# brand / chain
if "brand_name" in df.columns:
    df["brand_name_clean"] = df["brand_name"].astype(str).str.lower().replace("nan", np.nan)
else:
    df["brand_name_clean"] = np.nan

df["is_chain"] = df["brand_name_clean"].notna().astype(int)

brand_counts = df["brand_name_clean"].value_counts(dropna=True)
common_brands = set(brand_counts[brand_counts >= 5].index)

def brand_group(x):
    if pd.isna(x):
        return "independent"
    return x if x in common_brands else "other_chain"

df["brand_group"] = df["brand_name_clean"].apply(brand_group)

# type flags
important_types = [
    "restaurant", "cafe", "bar",
    "meal_takeaway", "meal_delivery",
    "bakery", "night_club", "store"
]

def types_to_set(s):
    if pd.isna(s):
        return set()
    return set(t.strip() for t in str(s).split(",") if t.strip())

type_sets = df["types"].apply(types_to_set) if "types" in df.columns else pd.Series([set()] * len(df))

for t in important_types:
    df[f"type_{t}"] = type_sets.apply(lambda st, t=t: int(t in st))

type_cols = [f"type_{t}" for t in important_types]

# ----- modelling frame -----
model_df = df[df["rating"].notna()].copy()

numeric_features = ["log_reviews", "price_level"] + type_cols
categorical_features = ["cuisine", "grid_id", "brand_group", "business_status"]

numeric_features = [c for c in numeric_features if c in model_df.columns]
categorical_features = [c for c in categorical_features if c in model_df.columns]

X = model_df[numeric_features + categorical_features]
y = model_df["rating"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]), numeric_features),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features),
    ],
    sparse_threshold=0
)

gbr = HistGradientBoostingRegressor(
    max_depth=6,
    learning_rate=0.05,
    max_iter=300,
    random_state=42
)

pipe = Pipeline(steps=[("prep", preprocess), ("model", gbr)])
pipe.fit(X, y)

model_df["expected_rating"] = pipe.predict(X)
model_df["hype_residual"] = model_df["rating"] - model_df["expected_rating"]

# restrict to places with enough reviews for stability
MIN_REVIEWS = 50
rank_df = model_df[model_df["user_ratings_total"] >= MIN_REVIEWS].copy()
output_path = os.path.join(args.output_dir, f"{args.city_name}_hype_adjusted_ratings.csv")
rank_df.to_csv(output_path, index=False)
print(f"Saved ratings to {output_path}")

# ============================================================
# 2. GRID-LEVEL HUB SCORE (PCA) + K-MEANS HUB TYPES
# ============================================================
def shannon_entropy(values):
    if len(values) == 0:
        return 0.0
    counts = Counter(values)
    total = sum(counts.values())
    probs = [c / total for c in counts.values() if c > 0]
    return -sum(p * log(p, 2) for p in probs)

df_rest = rank_df.copy()
df_rest = df_rest[df_rest["grid_id"].notna()].copy()
df_rest["user_ratings_total"] = pd.to_numeric(df_rest["user_ratings_total"], errors="coerce").fillna(0)
df_rest["log_reviews"] = np.log1p(df_rest["user_ratings_total"])
df_rest["is_chain"] = df_rest["is_chain"].fillna(0).astype(int)

if "price_level" in df_rest.columns:
    df_rest["price_level"] = pd.to_numeric(df_rest["price_level"], errors="coerce")
else:
    df_rest["price_level"] = np.nan

# aggregate by grid
agg = df_rest.groupby("grid_id").agg(
    n_places=("place_id", "count"),
    mean_rating=("rating", "mean"),
    median_rating=("rating", "median"),
    mean_hype_residual=("hype_residual", "mean"),
    total_reviews=("user_ratings_total", "sum"),
    mean_log_reviews=("log_reviews", "mean"),
    chain_share=("is_chain", "mean"),
    mean_price_level=("price_level", "mean"),
    centre_lat=("lat", "mean"),
    centre_lon=("lon", "mean")
).reset_index()

agg["independent_share"] = 1.0 - agg["chain_share"]

# cuisine entropy per grid
cuisine_entropy_list = []
for grid_id, sub in df_rest.groupby("grid_id"):
    cuisines = sub["cuisine"].fillna("unknown").astype(str).tolist()
    cuisine_entropy_list.append((grid_id, shannon_entropy(cuisines)))
entropy_df = pd.DataFrame(cuisine_entropy_list, columns=["grid_id", "cuisine_entropy"])
agg = agg.merge(entropy_df, on="grid_id", how="left")

hub_features = [
    "n_places",
    "mean_rating",
    "mean_hype_residual",
    "mean_log_reviews",
    "independent_share",
    "cuisine_entropy",
    "mean_price_level"
]

MIN_PLACES = 5
hub_df = agg[agg["n_places"] >= MIN_PLACES].copy()

if hub_df.empty:
    print("Warning: Not enough data for hub analysis (need grids with >= 5 places). Skipping hub analysis.")
    sys.exit(0)

X_hub = hub_df[hub_features].copy()
X_hub = X_hub.replace([np.inf, -np.inf], np.nan)
X_hub = X_hub.fillna(X_hub.median(numeric_only=True))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_hub)

# PCA -> hub_score
pca = PCA(n_components=1, random_state=42)
hub_df["hub_score_raw"] = pca.fit_transform(X_scaled).ravel()
corr = np.corrcoef(hub_df["hub_score_raw"], hub_df["mean_rating"])[0, 1]
hub_df["hub_score"] = hub_df["hub_score_raw"] * (1 if corr >= 0 else -1)

hub_df["hub_score_0_100"] = (
    (hub_df["hub_score"] - hub_df["hub_score"].min())
    / (hub_df["hub_score"].max() - hub_df["hub_score"].min())
    * 100
)

# K-means hub types
N_CLUSTERS = 4
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init="auto")
hub_df["hub_cluster_raw"] = kmeans.fit_predict(X_scaled)

cluster_order = (
    hub_df.groupby("hub_cluster_raw")["hub_score"]
          .mean()
          .sort_values(ascending=False)
          .reset_index()
)
cluster_order["hub_cluster_ranked"] = range(len(cluster_order))
mapping = dict(zip(cluster_order["hub_cluster_raw"], cluster_order["hub_cluster_ranked"]))
hub_df["hub_cluster"] = hub_df["hub_cluster_raw"].map(mapping)

cluster_names = {
    0: "Elite hubs",
    1: "Strong hubs",
    2: "Everyday hubs",
    3: "Weak hubs"
}
hub_df["hub_cluster_label"] = hub_df["hub_cluster"].map(cluster_names)

hub_output_path = os.path.join(args.output_dir, f"{args.city_name}_restaurant_hub_scores_with_clusters.csv")
hub_df.to_csv(hub_output_path, index=False)
print(f"Saved hub scores to {hub_output_path}")

# ============================================================
# 3. HEX REGIONS + BOROUGHS + TOP 5 HUBS
# ============================================================
# build point gdf from hub_df grid centroids
hub_df_map = hub_df.dropna(subset=["centre_lon", "centre_lat"]).copy()
gdf_web = gpd.GeoDataFrame(
    hub_df_map,
    geometry=gpd.points_from_xy(hub_df_map["centre_lon"], hub_df_map["centre_lat"]),
    crs="EPSG:4326"
).to_crs(epsg=3857)

xmin, ymin, xmax, ymax = gdf_web.total_bounds

# big hex grid (~3.5km)
hex_size = 3500  # metres

def make_hex_grid(xmin, ymin, xmax, ymax, size):
    hexes = []
    xs = np.arange(xmin - size, xmax + size, size * 3/2)
    ys = np.arange(ymin - size, ymax + size, size * np.sqrt(3))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            y_shift = y + (size * np.sqrt(3) / 2 if i % 2 else 0)
            coords = [
                (x + size * np.cos(theta), y_shift + size * np.sin(theta))
                for theta in np.linspace(0, 2*np.pi, 7)[:-1]
            ]
            hexes.append(Polygon(coords))
    return gpd.GeoDataFrame(geometry=hexes, crs=gdf_web.crs)

hex_gdf = make_hex_grid(xmin, ymin, xmax, ymax, hex_size)

# spatial join: assign each grid centroid to a hex
joined = gpd.sjoin(
    gdf_web[["hub_score_0_100", "hub_cluster_label", "geometry"]],
    hex_gdf.reset_index().rename(columns={"index": "hex_id"}),
    how="inner",
    predicate="within"
)

def mode_or_nan(series):
    if series.empty:
        return np.nan
    counts = Counter(series.dropna())
    return counts.most_common(1)[0][0] if counts else np.nan

hex_stats = (
    joined
    .groupby("hex_id")
    .agg(
        mean_hub_score=("hub_score_0_100", "mean"),
        n_tiles=("hub_score_0_100", "size"),
        dom_cluster=("hub_cluster_label", mode_or_nan)
    )
    .reset_index()
)

MIN_TILES_HEX = 3
hex_stats = hex_stats[hex_stats["n_tiles"] >= MIN_TILES_HEX].copy()

hex_gdf = (
    hex_gdf.reset_index().rename(columns={"index": "hex_id"})
    .merge(hex_stats, on="hex_id", how="inner")
)

print("Number of hexes after filtering:", len(hex_gdf))

# top 5 hubs (points) for labels
top5 = (
    hub_df.sort_values("hub_score", ascending=False)
          .head(5)
          .dropna(subset=["centre_lon", "centre_lat"])
          .copy()
)
top5["rank"] = range(1, len(top5) + 1)

top5_gdf = gpd.GeoDataFrame(
    top5,
    geometry=gpd.points_from_xy(top5["centre_lon"], top5["centre_lat"]),
    crs="EPSG:4326"
).to_crs(hex_gdf.crs)

# load borough boundaries
boroughs = gpd.read_file(PATH_BOROUGHS)
if boroughs.crs is None:
    boroughs.set_crs(epsg=4326, inplace=True)
boroughs = boroughs.to_crs(hex_gdf.crs)

# ============================================================
# 4. PLOTTING (FT-ish, with arrows + borough names)
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "text.color": "#333333",
    "axes.titlesize": 14,
    "figure.facecolor": "#f5f3ee",
    "axes.facecolor": "#f5f3ee"
})

ft_cmap = LinearSegmentedColormap.from_list(
    "ft_hub",
    ["#f6e7d7", "#f4a582", "#ca6d57", "#7b3254"]
)

cluster_palette = {
    "Elite hubs":     "#7b3254",
    "Strong hubs":    "#ca6d57",
    "Everyday hubs":  "#f4a582",
    "Weak hubs":      "#d8cbc0"
}

# ---------- attach borough names to top 5 hubs ----------
# guess the borough-name column
name_candidates = ["name", "NAME", "borough", "Borough", "BOROUGH", "lad22nm", "lad_name"]
name_col = None
for c in name_candidates:
    if c in boroughs.columns:
        name_col = c
        break

if name_col is not None:
    top5_with_boro = gpd.sjoin(
        top5_gdf,
        boroughs[[name_col, "geometry"]],
        how="left",
        predicate="within"
    )
    top5_gdf[name_col] = top5_with_boro[name_col].values
    top5_gdf["hub_label"] = np.where(
        top5_gdf[name_col].notna(),
        top5_gdf[name_col],
        "Hub " + top5_gdf["rank"].astype(int).astype(str)
    )
else:
    # fallback: just use "Hub 1" etc.
    top5_gdf["hub_label"] = "Hub " + top5_gdf["rank"].astype(int).astype(str)

# Manual override for Hub 4
top5_gdf.loc[top5_gdf["rank"] == 4, "hub_label"] = "Ealing-Acton"

# simple helper: arrow offsets spread around the map
dx = xmax - xmin
dy = ymax - ymin
arrow_offsets = [
    (+0.04 * dx, +0.03 * dy),
    (-0.05 * dx, +0.02 * dy),
    (+0.05 * dx, -0.02 * dy),
    (-0.04 * dx, -0.03 * dy),
    (+0.03 * dx, +0.05 * dy),
]

# ---------- Combined Figure ----------
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
# Reduced bottom margin to decrease whitespace
plt.subplots_adjust(wspace=0.05, bottom=0.12, top=0.92, left=0.02, right=0.98)

# --- Left Plot: London's Restaurant Hubs ---
ax0 = axes[0]

# borough outlines
boroughs.boundary.plot(
    ax=ax0,
    linewidth=0.6,
    edgecolor="#bbbbbb",
    alpha=0.9,
    zorder=1
)

# Calculate robust limits for better color contrast
vmin = np.percentile(hex_gdf["mean_hub_score"], 5)
vmax = np.percentile(hex_gdf["mean_hub_score"], 95)

# hex polygons
hex_plot = hex_gdf.plot(
    ax=ax0,
    column="mean_hub_score",
    cmap=ft_cmap,
    linewidth=0.4,
    edgecolor="#f5f3ee",
    alpha=0.9,
    zorder=2,
    vmin=vmin,
    vmax=vmax
)

# top 5 hub stars + arrows with labels
for i, (_, row) in enumerate(top5_gdf.iterrows()):
    x, y = row.geometry.x, row.geometry.y
    offx, offy = arrow_offsets[i % len(arrow_offsets)]

    # marker
    ax0.scatter(
        x, y,
        s=80,
        marker="D",
        color="#222222",
        edgecolor="#ffffff",
        linewidth=0.8,
        zorder=3
    )

    # arrow + label
    ax0.annotate(
        row["hub_label"],
        xy=(x, y),
        xytext=(x + offx, y + offy),
        fontsize=8,
        ha="center",
        va="center",
        arrowprops=dict(
            arrowstyle="->",
            linewidth=0.6,
            color="#444444"
        ),
        zorder=4
    )

ax0.set_xlim(xmin, xmax)
ax0.set_ylim(ymin, ymax)
ax0.set_axis_off()
ax0.set_title("London’s Restaurant Hubs", loc="left", pad=12, fontweight="bold")

# Colorbar for left plot
norm = Normalize(vmin=vmin, vmax=vmax)
sm = ScalarMappable(cmap=ft_cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax0, shrink=0.6, pad=0.04, location='bottom')
cbar.ax.set_xlabel("Restaurant Hub Score (0–100)", labelpad=5)


# --- Right Plot: Types of Restaurant Hubs ---
ax1 = axes[1]

# Prepare cluster regions
hex_gdf["dom_cluster"] = hex_gdf["dom_cluster"].fillna("Weak hubs")
cluster_regions = hex_gdf.dissolve(by="dom_cluster", as_index=False)

boroughs.boundary.plot(
    ax=ax1,
    linewidth=0.6,
    edgecolor="#bbbbbb",
    alpha=0.9,
    zorder=1
)

for _, row in cluster_regions.iterrows():
    color = cluster_palette.get(row["dom_cluster"], "#d8cbc0")
    gpd.GeoSeries([row.geometry], crs=cluster_regions.crs).plot(
        ax=ax1,
        facecolor=color,
        edgecolor="none",
        alpha=0.75,
        zorder=2
    )

# stars + arrows again
for i, (_, row) in enumerate(top5_gdf.iterrows()):
    x, y = row.geometry.x, row.geometry.y
    offx, offy = arrow_offsets[i % len(arrow_offsets)]

    ax1.scatter(
        x, y,
        s=80,
        marker="D",
        color="#222222",
        edgecolor="#ffffff",
        linewidth=0.8,
        zorder=3
    )

    ax1.annotate(
        row["hub_label"],
        xy=(x, y),
        xytext=(x + offx, y + offy),
        fontsize=8,
        ha="center",
        va="center",
        arrowprops=dict(
            arrowstyle="->",
            linewidth=0.6,
            color="#444444"
        ),
        zorder=4
    )

ax1.set_xlim(xmin, xmax)
ax1.set_ylim(ymin, ymax)
ax1.set_axis_off()
ax1.set_title("Types of Restaurant Hubs", loc="left", pad=12, fontweight="bold")

# Dummy colorbar for right plot to ensure equal sizing
sm_dummy = ScalarMappable(cmap=ft_cmap, norm=norm)
sm_dummy.set_array([])
cbar_dummy = fig.colorbar(sm_dummy, ax=ax1, shrink=0.6, pad=0.04, location='bottom')
cbar_dummy.ax.set_visible(False)

# legend for cluster regions
handles, labels = [], []
for lab, col in cluster_palette.items():
    if lab in cluster_regions["dom_cluster"].unique():
        h = plt.Line2D(
            [], [], marker="s", linestyle="",
            markersize=8, markerfacecolor=col, markeredgecolor=col
        )
        handles.append(h)
        labels.append(lab)

ax1.legend(
    handles, labels,
    title="Dominant hub type",
    loc="lower right",
    frameon=False,
    fontsize=9,
    title_fontsize=9,
    bbox_to_anchor=(1.0, 0.0)
)

# --- Shared Description ---
fig.text(
    0.02, 0.06,
    "Left: Hexagons aggregate nearby restaurant tiles. Higher scores combine density, ratings, surprise, cuisine diversity and independent share.\n"
    "Right: Merged regions show where each hub type is dominant. Arrows highlight and label the top 5 hubs by hub score.",
    ha="left",
    va="top",
    fontsize=10,
    color="#555555"
)
fig.text(
    0.02, 0.01,
    "Source: Google Maps scrape; author’s calculations",
    ha="left",
    va="bottom",
    fontsize=9,
    color="#777777"
)

plt.show()

# ============================================================
# 5. PANEL MAPS: TOP 10 CUISINES DENSITY
# ============================================================
print("Generating cuisine density panel...")

# 1. Identify top 10 cuisines (excluding 'unknown' and 'cafe')
excluded_cuisines = ["unknown", "cafe"]
top_10_cuisines = (
    df[~df["cuisine"].isin(excluded_cuisines)]["cuisine"]
    .value_counts()
    .head(10)
    .index.tolist()
)

# 2. Create GeoDataFrame for all restaurants
# Ensure lat/lon are numeric and available
# (Assuming lat/lon are in df as per previous sections, but checking just in case)
if "lat" not in df.columns or "lon" not in df.columns:
    # Try to recover from df_basic if needed
    temp_basic = df_basic[["place_id", "lat", "lon"]].copy()
    df_map = df.merge(temp_basic, on="place_id", how="left", suffixes=("", "_basic"))
    if "lat" in df_map.columns:
        df_map["lat"] = df_map["lat"].fillna(df_map.get("lat_basic"))
        df_map["lon"] = df_map["lon"].fillna(df_map.get("lon_basic"))
    else:
        df_map["lat"] = df_map["lat_basic"]
        df_map["lon"] = df_map["lon_basic"]
else:
    df_map = df.copy()

df_map = df_map.dropna(subset=["lat", "lon"])

gdf_restaurants = gpd.GeoDataFrame(
    df_map,
    geometry=gpd.points_from_xy(df_map["lon"], df_map["lat"]),
    crs="EPSG:4326"
).to_crs(hex_gdf.crs)

# 3. Re-create full hex grid to ensure we cover the whole area (including low-density areas)
# We reuse the bounds and size from Section 3
full_hex_gdf = make_hex_grid(xmin, ymin, xmax, ymax, hex_size)
full_hex_gdf["hex_id"] = range(len(full_hex_gdf))

# Spatial join
joined_cuisines = gpd.sjoin(
    gdf_restaurants,
    full_hex_gdf,
    how="inner",
    predicate="within"
)

# 4. Setup plot
fig, axes = plt.subplots(2, 5, figsize=(18, 8), constrained_layout=True)
axes = axes.flatten()

# FT-ish palette for the panels
panel_colors = [
    "#7b3254", # Claret
    "#2e6e9e", # Blue
    "#e68a00", # Orange
    "#0f544e", # Dark Green
    "#ca6d57", # Terracotta
    "#5d6d7e", # Slate
    "#8e44ad", # Purple
    "#d35400", # Pumpkin
    "#27ae60", # Green
    "#c0392b"  # Red
]

for i, cuisine in enumerate(top_10_cuisines):
    ax = axes[i]
    color = panel_colors[i % len(panel_colors)]
    
    # Make river visible: set background to water color, boroughs to land color
    ax.set_facecolor("#cce3f0")  # Light blue water
    
    boroughs.plot(
        ax=ax,
        facecolor="#f5f3ee",     # Land color
        linewidth=0.4,
        edgecolor="#bbbbbb",
        alpha=1.0,
        zorder=1
    )
    
    # Data: count per hex for this cuisine
    subset = joined_cuisines[joined_cuisines["cuisine"] == cuisine]
    if not subset.empty:
        counts = subset.groupby("hex_id").size().reset_index(name="n")
        plot_gdf = full_hex_gdf.merge(counts, on="hex_id", how="inner")
        
        # Plot density
        # We use a custom colormap from transparent/light to the target color
        cmap = LinearSegmentedColormap.from_list(f"cmap_{i}", ["#f5f3ee", color])
        
        plot_gdf.plot(
            ax=ax,
            column="n",
            cmap=cmap,
            linewidth=0.1,
            edgecolor=color,
            alpha=0.9,
            zorder=2
        )
    
    ax.set_title(cuisine.title(), loc="left", fontsize=11, fontweight="bold", color="#333333")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_axis_off()

fig.suptitle("Density of Top 10 Cuisines in London", fontsize=18, fontweight="bold", x=0.01, ha="left")
fig.text(
    0.01, 0.02,
    "Source: Google Maps scrape; author’s calculations. Color intensity represents number of restaurants per hex.",
    ha="left",
    fontsize=10,
    color="#777777"
)

plt.show()


