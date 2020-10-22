# %% read csv file
from datetime import datetime
from matplotlib.pyplot import xlabel
from numpy.lib.function_base import _median
import pandas as pd
import seaborn as sns
import numpy as np

df = pd.read_csv("2012.csv")

#%%Describe the meaning and type of data (e.g., scale, values) 
# for each attribute in the data file

print(df.shape)
print(df.columns)
print(df.info())
print(df.isna())
print(df.isna().sum())
print(df.isnull().any())
print(df.isna().any()) # axis 0 by default . It acts on all rows
print (df[df.isnull().any(axis=1)].head())# axis 1 gives you 
#pd.set_option("max_rows", None)- Used to show all in a row.
#pd.set_option('max_columns', None)- Used to show all columns
#pd.reset_option("max_rows")- to reset to default


#%% Ensure all missing values are captured as NaN

missing_value_formats = ["n.a.","?","NA","n/a", "na", "--"]
df = pd.read_csv("2012.csv", na_values = missing_value_formats)


# %% make sure these number columns contain numbers only
# if a value cannot be converted to a number, make it NaN
from tqdm import tqdm

cols = [
    "perobs",
    "perstop",
    "age",
    "weight",
    "ht_feet",
    "ht_inch",
    "datestop",
    "timestop",
    "xcoord",
    "ycoord",
]
for col in tqdm(cols, desc="convert to number"):
    df[col] = pd.to_numeric(df[col], errors="coerce")


# %% drop rows with invalid numberic value
df = df.dropna()

#%%Determine outliers using Boxplot
from scipy import stats
import matplotlib.pyplot as plt

sns.boxplot(x=df["ht_feet"])
#%%
sns.boxplot(x=df["weight"])
#%%
sns.boxplot(x=df["age"])
#%%
sns.boxplot(x=df["perobs"])

#%%
sns.boxplot(x=df["year"])

#%% Handling age outliers
import pandas as pd
median = df.loc[df['age']<10, 'age'].median()
df.loc[df.age > 100, 'age'] = np.nan
df.fillna(median,inplace=True)

#%% Handling weight outliers
median = df.loc[df['weight']<50, 'weight'].median()
df.loc[df.weight > 350, 'age'] = np.nan
df.fillna(median,inplace=True)

#%% Using Isolation forest for outliers
#from sklearn.ensemble import IsolationForest
#np.random.seed(1)
#df = np.random.randn(516985,112)  * 20 + 20

#clf = IsolationForest( behaviour = 'new', max_samples=100, random_state = 1, contamination= 'auto')
#preds = clf.fit_predict(df)
#preds

#print (preds < 0)
#print(np.count_nonzero(preds < 0))
#gives the number of outlier data points


#%% Using the Z score to determine outliers

#from scipy import stats
#z=np.abs(stats.zscore(df))
#print(z)

#%%Using 3 and -3 as the threshold remove outliers

#print(np.where(z>3)) 

#%% 
#print(np.where(z<-3))
#%% Handling outliers

#df_z=df[(z<3).all(axis=1)]



# %% show stats
df.describe(include="all")



# %% make datetime column
df["datestop"] = df["datestop"].astype(str).str.zfill(8)
df["timestop"] = df["timestop"].astype(str).str.zfill(4)

from datetime import datetime


def make_datetime(datestop, timestop):
    year = int(datestop[-4:])
    month = int(datestop[:2])
    day = int(datestop[2:4])

    hour = int(timestop[:2])
    minute = int(timestop[2:])

    return datetime(year, month, day, hour, minute)


df["datetime"] = df.apply(
    lambda row: make_datetime(row["datestop"], row["timestop"]), axis=1
)


# %% make height column
df["height"] = (df["ht_feet"] * 12 + df["ht_inch"]) * 2.54


# %% make lat/lon columns
import pyproj

srs = (
    "+proj=lcc +lat_1=41.03333333333333 "
    "+lat_2=40.66666666666666 +lat_0=40.16666666666666 +lon_0=-74 "
    "+x_0=300000.0000000001 +y_0=0 "
    "+ellps=GRS80 +datum=NAD83 +to_meter=0.3048006096012192 +no_defs"
)

p = pyproj.Proj(srs)

coords = df.apply(lambda r: p(r["xcoord"], r["ycoord"], inverse=True), axis=1)
df["lat"] = [c[1] for c in coords]
df["lon"] = [c[0] for c in coords]


# %% read the spec file and replace values in df with the matching labels
import numpy as np

value_labels = pd.read_excel(
    "2012 SQF File Spec.xlsx", sheet_name="Value Labels", skiprows=range(4)
)
value_labels["Field Name"] = value_labels["Field Name"].fillna(method="ffill")
value_labels["Field Name"] = value_labels["Field Name"].str.lower()
value_labels["Value"] = value_labels["Value"].fillna(" ")
vl_mapping = value_labels.groupby("Field Name").apply(
    lambda x: dict([(row["Value"], row["Label"]) for row in x.to_dict("records")])
)

cols = [col for col in df.columns if col in vl_mapping]

for col in tqdm(cols):
    df[col] = df[col].apply(lambda val: vl_mapping[col].get(val, np.nan))


# %% plot height
import seaborn as sns

sns.distplot(df["height"])


# %% plot month
sns.countplot(df["datetime"].dt.month)


# %% plot day of week
ax = sns.countplot(df["datetime"].dt.weekday)
ax.set_xticklabels(["Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun"])
ax.set(xlabel="day of week", title="# of incidents by day of weeks")
ax.get_figure().savefig("test.png")


# %% export stats to excel
df.describe(include="all").to_excel("stats.xlsx")


# %% plot hour
ax = sns.countplot(df["datetime"].dt.hour)


# %% plot xcoord / ycoord
sns.scatterplot(data=df[:100], x="xcoord", y="ycoord")

# %% plot lat / lon of murder cases on an actual map
import folium

m = folium.Map((40.7128, -74.0060))

for r in df[["lat", "lon"]][df["detailcm"] == "MURDER"].to_dict("records"):
    folium.CircleMarker(location=(r["lat"], r["lon"]), radius=1).add_to(m)

m

# %% plot lat / lon of terrorism cases on an actual map

m = folium.Map((40.7128, -74.0060))

for r in df[["lat", "lon"]][df["detailcm"] == "TERRORISM"].to_dict("records"):
    folium.CircleMarker(location=(r["lat"], r["lon"]), radius=1).add_to(m)

m

# %% plot race
sns.countplot(data=df, y="race")


# %% plot race wrt city
sns.countplot(data=df, y="race", hue="city")


# %% plot top crimes where physical forces used
pf_used = df[
    (df["pf_hands"] == "YES")
    | (df["pf_wall"] == "YES")
    | (df["pf_grnd"] == "YES")
    | (df["pf_drwep"] == "YES")
    | (df["pf_ptwep"] == "YES")
    | (df["pf_baton"] == "YES")
    | (df["pf_hcuff"] == "YES")
    | (df["pf_pepsp"] == "YES")
    | (df["pf_other"] == "YES")
]

sns.countplot(
    data=pf_used,
    y="detailcm",
    order=pf_used["detailcm"].value_counts(ascending=False).keys()[:10],
)


# %% plot percentage of each physical forces used
pfs = [col for col in df.columns if col.startswith("pf_")]
pf_counts = (df[pfs] == "YES").sum()
sns.barplot(y=pf_counts.index, x=pf_counts.values / df.shape[0])


# %% drop columns that are not used in analysis
df = df.drop(
    columns=[
        # processed columns
        "datestop",
        "timestop",
        "ht_feet",
        "ht_inch",
        "xcoord",
        "ycoord",
        # not useful
        "year",
        "recstat",
        "crimsusp",
        "dob",
        "ser_num",
        "arstoffn",
        "sumoffen",
        "compyear",
        "comppct",
        "othfeatr",
        "adtlrept",
        "dettypcm",
        "linecm",
        "repcmd",
        "revcmd",
        # location of stop
        # only use coord and city
        "addrtyp",
        "rescode",
        "premtype",
        "premname",
        "addrnum",
        "stname",
        "stinter",
        "crossst",
        "aptnum",
        "state",
        "zip",
        "addrpct",
        "sector",
        "beat",
        "post",
    ]
)

# %% modify one column
df["trhsloc"] = df["trhsloc"].fillna("NEITHER")

# %% remove all rows with NaN
df = df.dropna()

# %% save dataframe to a file
df.to_pickle("data.pkl")
