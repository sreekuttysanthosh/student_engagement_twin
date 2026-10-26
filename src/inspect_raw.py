import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path, sep="\t", low_memory=False)

print("Shape:", df.shape)
print("\nAll columns, dtypes, % missing:")
summary = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "pct_missing": (df.isna().mean() * 100).round(2),
    "n_unique": df.nunique(),
})
print(summary.to_string())

print("\nColumns we care about:")
for c in ["nevents", "ndays_act", "ncontent", "nforum_posts", "completed_%"]:
    if c in df.columns:
        print(f"  {c}: missing {df[c].isna().mean():.2%}, "
              f"min {df[c].min()}, max {df[c].max()}")
    else:
        print(f"  {c}: NOT FOUND")

print("\nFirst 3 rows:")
print(df.head(3).T.to_string())