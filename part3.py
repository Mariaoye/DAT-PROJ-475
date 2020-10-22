# %% read dataframe
import pandas as pd

df = pd.read_pickle("data.pkl")


#%%

df = df.drop(
    columns=[
        # processed columns
        "pf_ptwep",
        "pf_hands",
        "pf_wall",
        "pf_grnd",
        "pf_drwep",
        "pf_ptwep",
        "pf_baton",
        "pf_hcuff",
        "pf_pepsp",
        "pf_other",
        "radio",
        "ac_rept",
        "ac_inves",
        "rf_vcrim",
        "rf_othsw",
        "ac_proxm",
        "ac_evasv",
        "ac_assoc",
        "ac_cgdir",
        "rf_verbl",
        "ac_incid",
        "ac_time",
        "ac_stsnd",
        "ac_other",
        "rf_furt",
        "rf_bulg",
    ]
)

#%% New data frame
df.shape
df.info
# %% pick a crime
df_assault = df[df["detailcm"] == "ASSAULT"]

# %% apply hierarchical clustering on a range of arbitrary values
# record the silhouette_score and find the best number of clusters

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from tqdm import tqdm

scores, labels = {}, {}
num_city = df["city"].nunique()
num_pct = df["pct"].nunique()


for k in tqdm(range(num_city, num_pct, 10)):
    c = AgglomerativeClustering(n_clusters=k)
    y = c.fit_predict(df_assault[["lat", "lon"]])
    scores[k] = silhouette_score(df_assault[["lat", "lon"]], y)
    labels[k] = y

# %% find the best k visually
import seaborn as sns

sns.lineplot(x=scores.keys(), y=scores.values())


# %% find the best k by code
best_k = max(scores, key=lambda k: scores[k])


# %% visualize the hierarchcal clustering result
import folium

m = folium.Map((40.7128, -74.0060))
colors = sns.color_palette("hls", best_k).as_hex()
df_assault["label"] = labels[best_k]
for r in df_assault.to_dict("records"):
    folium.CircleMarker(
        location=(r["lat"], r["lon"]), radius=1, color=colors[r["label"]]
    ).add_to(m)

m
# %% CRIMESTOP
# %% find reason for stop columns
# and apply dbscan
from sklearn.cluster import DBSCAN

css = [col for col in df.columns if col.startswith("cs_")]
c = DBSCAN()
x = df_assault[css] == "YES"
y = c.fit_predict(x)
print(silhouette_score(x, y))

#Note: you can use kmeans or hierarchical clustering as well


# %% visualize the result on map
import numpy as np

m = folium.Map((40.7128, -74.0060))
best_k = len(np.unique(y))
colors = sns.color_palette("hls", best_k).as_hex()
df_assault["label"] = y
for r in df_assault.to_dict("records"):
    folium.CircleMarker(
        location=(r["lat"], r["lon"]), radius=1, color=colors[r["label"]]
    ).add_to(m)

m

#Then, you can perform a value_counts and see the # of appearances for each label  

# %%
df_assault["label"].value_counts()


# %% pick a label and visualize the datapoints on map
biggest_cluster = df_assault["label"].value_counts().index[0]
m = folium.Map((40.7128, -74.0060))
for r in df_assault[df_assault["label"] == biggest_cluster].to_dict("records"):
    folium.CircleMarker(
        location=(r["lat"], r["lon"]), radius=1, color=colors[r["label"]]
    ).add_to(m)

m


# %%
#Cluster arrest made

df_arst = df[df["arstmade"] == "YES"]

# %% apply hierarchical clustering on a range of arbitrary values
# record the silhouette_score and find the best number of clusters

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from tqdm import tqdm

scores, labels = {}, {}
num_city = df["city"].nunique()
num_pct = df["pct"].nunique()


for k in tqdm(range(num_city, num_pct, 10)):
    c = AgglomerativeClustering(n_clusters=k)
    y = c.fit_predict(df_arst[["lat", "lon"]])
    scores[k] = silhouette_score(df_arst[["lat", "lon"]], y)
    labels[k] = y

# %% find the best k visually
import seaborn as sns

sns.lineplot(x=scores.keys(), y=scores.values())


# %% find the best k by code
best_k = max(scores, key=lambda k: scores[k])


# %% visualize the hierarchcal clustering result
import folium

m = folium.Map((40.7128, -74.0060))
colors = sns.color_palette("hls", best_k).as_hex()
df_arst["label"] = labels[best_k]
for r in df_arst.to_dict("records"):
    folium.CircleMarker(
        location=(r["lat"], r["lon"]), radius=1, color=colors[r["label"]]
    ).add_to(m)

m
# %%
