import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, os

warnings.filterwarnings("ignore")
np.random.seed(42)
sns.set_theme(style="whitegrid", palette="muted")

OUT_DIR = "data"
FIG_DIR = "reports"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. DATASET SCHEMA
# ─────────────────────────────────────────────
SCHEMA = {
    "intern_id":              "Unique identifier",
    "department":             "Team/department assigned",
    "duration_weeks":         "Internship length (4-24 weeks)",
    "tasks_assigned":         "Total tasks given",
    "tasks_completed":        "Tasks successfully finished",
    "task_completion_rate":   "tasks_completed / tasks_assigned",
    "avg_task_quality":       "Peer/mentor quality score (1-10)",
    "on_time_delivery_rate":  "Fraction of tasks submitted on time",
    "days_absent":            "Absences during internship",
    "attendance_rate":        "Attendance fraction",
    "meetings_attended":      "Internal meetings attended",
    "meeting_attendance_rate":"meetings_attended / total_meetings",
    "feedback_score":         "Mentor feedback (1-10)",
    "peer_collab_score":      "Peer collaboration rating (1-10)",
    "initiative_score":       "Self-started work rating (1-10)",
    "communication_score":    "Communication rating (1-10)",
    "learning_curve_score":   "Skill growth assessment (1-10)",
    "tools_used":             "Number of distinct tools/tech used",
    "bugs_introduced":        "Bugs logged (lower = better)",
    "code_review_pass_rate":  "Code accepted on first review",
    "prior_experience_years": "Years of relevant prior experience",
    "performance_label":      "Target: High / Medium / Low",
}

print("=" * 60)
print("  SPRINT 1 — DATA FOUNDATION")
print("=" * 60)
print(f"\n📋 Dataset Schema ({len(SCHEMA)} columns):")
for col, desc in SCHEMA.items():
    print(f"   {col:<30} → {desc}")

# ─────────────────────────────────────────────
# 2. SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────
N = 500
DEPARTMENTS = ["Engineering", "Data Science", "Design", "Marketing", "Product"]

def generate_intern(i):
    dept  = np.random.choice(DEPARTMENTS)
    weeks = np.random.randint(4, 25)

    # Core metrics — with realistic correlations
    base_skill   = np.random.beta(2, 2)          # 0-1 latent skill
    base_engage  = np.random.beta(2, 2)          # 0-1 latent engagement

    tasks_assigned   = np.random.randint(10, 60)
    tasks_completed  = int(tasks_assigned * np.clip(base_skill * 1.2 + np.random.normal(0, 0.1), 0, 1))
    tcr              = tasks_completed / tasks_assigned

    quality          = round(np.clip(base_skill * 8 + np.random.normal(0, 0.8) + 1, 1, 10), 1)
    on_time          = round(np.clip(base_engage * 0.7 + base_skill * 0.3 + np.random.normal(0, 0.05), 0, 1), 2)

    total_meetings   = int(weeks * 2.5)
    meetings_att     = int(total_meetings * np.clip(base_engage + np.random.normal(0, 0.08), 0, 1))
    mar              = round(meetings_att / total_meetings, 2)

    days_absent      = int(np.clip(np.random.poisson((1 - base_engage) * 5), 0, weeks * 5))
    attendance       = round(np.clip(1 - days_absent / (weeks * 5), 0, 1), 2)

    feedback         = round(np.clip((base_skill * 0.6 + base_engage * 0.4) * 8 + np.random.normal(0, 0.7) + 1, 1, 10), 1)
    collab           = round(np.clip(base_engage * 8 + np.random.normal(0, 1) + 1, 1, 10), 1)
    initiative       = round(np.clip(base_skill * 5 + base_engage * 3 + np.random.normal(0, 0.8) + 1, 1, 10), 1)
    communication    = round(np.clip(base_engage * 7 + np.random.normal(0, 1) + 2, 1, 10), 1)
    learning         = round(np.clip(base_skill * 9 + np.random.normal(0, 0.5) + 0.5, 1, 10), 1)

    tools_used       = int(np.clip(weeks / 3 + base_skill * 5 + np.random.randint(0, 4), 1, 15))
    bugs             = int(np.clip(np.random.poisson((1 - base_skill) * 8), 0, 20))
    crpr             = round(np.clip(base_skill * 0.6 + 0.3 + np.random.normal(0, 0.05), 0, 1), 2)

    prior_exp        = round(np.clip(np.random.exponential(0.8), 0, 5), 1)

    # ── Performance label ──────────────────────────
    composite = (
        tcr          * 0.20 +
        quality/10   * 0.15 +
        feedback/10  * 0.15 +
        on_time      * 0.10 +
        collab/10    * 0.10 +
        initiative/10* 0.10 +
        learning/10  * 0.10 +
        attendance   * 0.10
    )
    if composite >= 0.72:
        label = "High"
    elif composite >= 0.50:
        label = "Medium"
    else:
        label = "Low"

    return {
        "intern_id":              f"INT-{i:04d}",
        "department":             dept,
        "duration_weeks":         weeks,
        "tasks_assigned":         tasks_assigned,
        "tasks_completed":        tasks_completed,
        "task_completion_rate":   round(tcr, 2),
        "avg_task_quality":       quality,
        "on_time_delivery_rate":  on_time,
        "days_absent":            days_absent,
        "attendance_rate":        attendance,
        "meetings_attended":      meetings_att,
        "meeting_attendance_rate":mar,
        "feedback_score":         feedback,
        "peer_collab_score":      collab,
        "initiative_score":       initiative,
        "communication_score":    communication,
        "learning_curve_score":   learning,
        "tools_used":             tools_used,
        "bugs_introduced":        bugs,
        "code_review_pass_rate":  crpr,
        "prior_experience_years": prior_exp,
        "performance_label":      label,
    }

df_raw = pd.DataFrame([generate_intern(i) for i in range(1, N + 1)])
print(f"\n✅ Generated {len(df_raw)} synthetic intern records.")
print(f"   Shape: {df_raw.shape}")

# ─────────────────────────────────────────────
# 3. DATA CLEANING
# ─────────────────────────────────────────────
print("\n🔧 Data Cleaning Pipeline:")

# 3a. Inject 2% nulls for realism
null_cols = ["avg_task_quality", "feedback_score", "code_review_pass_rate", "peer_collab_score"]
for col in null_cols:
    mask = np.random.choice([True, False], size=N, p=[0.02, 0.98])
    df_raw.loc[mask, col] = np.nan

print(f"   Missing values injected: {df_raw.isnull().sum().sum()}")

# 3b. Fill nulls — median for numerics
num_cols = df_raw.select_dtypes(include=np.number).columns
for col in num_cols:
    if df_raw[col].isnull().any():
        df_raw[col] = df_raw[col].fillna(df_raw[col].median())

# 3c. Inject & remove duplicates
df_raw = pd.concat([df_raw, df_raw.sample(5)], ignore_index=True)
before = len(df_raw)
df_raw.drop_duplicates(inplace=True)
df_raw.reset_index(drop=True, inplace=True)
print(f"   Duplicates removed: {before - len(df_raw)}")

# 3d. Clip outliers (IQR × 1.5)
clip_cols = ["bugs_introduced", "days_absent", "tasks_assigned"]
for col in clip_cols:
    Q1, Q3 = df_raw[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df_raw[col] = df_raw[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

# 3e. Validate ranges
assert df_raw["task_completion_rate"].between(0, 1).all()
assert df_raw["attendance_rate"].between(0, 1).all()
print(f"   Validation passed ✓")
print(f"   Final dataset: {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")
print(f"   Missing after clean: {df_raw.isnull().sum().sum()}")

# Class distribution
dist = df_raw["performance_label"].value_counts()
print(f"\n   Label distribution:\n{dist.to_string()}")

df_raw.to_csv(f"{OUT_DIR}/intern_data_clean.csv", index=False)
print(f"\n💾 Saved → {OUT_DIR}/intern_data_clean.csv")

# ─────────────────────────────────────────────
# 4. EDA VISUALIZATIONS (combined figure)
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16))
fig.suptitle("Sprint 1 — Intern Dataset EDA", fontsize=18, fontweight="bold", y=0.98)
fig.patch.set_facecolor("#F8F9FA")

colors_label = {"High": "#2ECC71", "Medium": "#F39C12", "Low": "#E74C3C"}

# (A) Label distribution
ax1 = fig.add_subplot(3, 4, 1)
counts = df_raw["performance_label"].value_counts()
bars = ax1.bar(counts.index, counts.values,
               color=[colors_label[l] for l in counts.index], edgecolor="white", linewidth=1.5)
for b in bars:
    ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 3,
             str(int(b.get_height())), ha="center", fontsize=10, fontweight="bold")
ax1.set_title("Performance Label Distribution", fontweight="bold")
ax1.set_ylabel("Count")

# (B) Department distribution
ax2 = fig.add_subplot(3, 4, 2)
dept_counts = df_raw["department"].value_counts()
ax2.barh(dept_counts.index, dept_counts.values, color="#5B86E5", edgecolor="white")
ax2.set_title("Interns per Department", fontweight="bold")

# (C) Task completion rate by label
ax3 = fig.add_subplot(3, 4, 3)
for label, grp in df_raw.groupby("performance_label"):
    ax3.hist(grp["task_completion_rate"], bins=20, alpha=0.65,
             label=label, color=colors_label[label])
ax3.set_title("Task Completion Rate by Label", fontweight="bold")
ax3.set_xlabel("Rate")
ax3.legend()

# (D) Feedback score distribution
ax4 = fig.add_subplot(3, 4, 4)
for label, grp in df_raw.groupby("performance_label"):
    ax4.hist(grp["feedback_score"], bins=15, alpha=0.65,
             label=label, color=colors_label[label])
ax4.set_title("Feedback Score by Label", fontweight="bold")
ax4.set_xlabel("Score")
ax4.legend()

# (E) Correlation heatmap
ax5 = fig.add_subplot(3, 4, (5, 6))
numeric_df = df_raw.select_dtypes(include=np.number)
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax5, cmap="RdYlGn", center=0,
            annot=False, linewidths=0.3, cbar_kws={"shrink": 0.7})
ax5.set_title("Feature Correlation Matrix", fontweight="bold")
ax5.tick_params(axis="x", rotation=45, labelsize=7)
ax5.tick_params(axis="y", labelsize=7)

# (F) Boxplots: key metrics by label
key_metrics = ["task_completion_rate", "avg_task_quality", "feedback_score", "initiative_score"]
for idx, metric in enumerate(key_metrics, start=7):
    ax = fig.add_subplot(3, 4, idx)
    data_plot = [df_raw[df_raw["performance_label"] == l][metric].values
                 for l in ["High", "Medium", "Low"]]
    bp = ax.boxplot(data_plot, patch_artist=True,
                    boxprops=dict(linewidth=1.5),
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors_label.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(["High", "Medium", "Low"])
    ax.set_title(metric.replace("_", " ").title(), fontweight="bold", fontsize=9)

# (G) Scatter: quality vs completion colored by label
ax_last = fig.add_subplot(3, 4, (11, 12))
for label, grp in df_raw.groupby("performance_label"):
    ax_last.scatter(grp["task_completion_rate"], grp["avg_task_quality"],
                    alpha=0.45, s=20, label=label, color=colors_label[label])
ax_last.set_xlabel("Task Completion Rate")
ax_last.set_ylabel("Avg Task Quality")
ax_last.set_title("Quality vs Completion Rate", fontweight="bold")
ax_last.legend(markerscale=2)

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{FIG_DIR}/sprint1_eda.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"\n📊 EDA figure saved → {FIG_DIR}/sprint1_eda.png")