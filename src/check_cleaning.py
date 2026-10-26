import pandas as pd

path = "data/raw/CNPC_1401-1509_DI_v1_1_2016-03-01.tab"
df = pd.read_csv(path, sep="\t", low_memory=False)

clean = df[df["completed_%"].notna()].copy()
print("Rows after keeping non-missing completed_%:", len(clean))
print("Paper reports: 26661")

print("\nFeature missingness in this subset (paper: 17.08, 10.44, 0, 68.05):")
for c in ["nevents", "ndays_act", "ncontent", "nforum_posts"]:
    print(f"  {c}: {clean[c].isna().mean()*100:.2f}%")

print("\ncompleted_% summary:")
print(clean["completed_%"].describe().to_string())

clean["Engagement_Level"] = (clean["completed_%"] >= 0.5).astype(int)
print("\nTarget distribution (0 = <50%, 1 = >=50%):")
print(clean["Engagement_Level"].value_counts().sort_index().to_string())
print(clean["Engagement_Level"].value_counts(normalize=True).sort_index().round(4).to_string())

print("\nDuplicate (user, course) pairs:",
      clean.duplicated(["userid_DI", "course_id_DI"]).sum())
print("Courses in subset:", clean["course_id_DI"].nunique())
