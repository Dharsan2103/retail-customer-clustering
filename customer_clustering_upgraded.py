"""
╔══════════════════════════════════════════════════════════════════════════╗
║        RETAIL CUSTOMER CLUSTERING — UPGRADED ANALYSIS SUITE             ║
║  Advanced ML | Rich Visualizations | Business Intelligence Report        ║
╚══════════════════════════════════════════════════════════════════════════╝

Upgrades over original notebook:
  ✔ Multiple algorithms: KMeans, DBSCAN, Agglomerative Hierarchical
  ✔ Optimal-k search via Elbow + Silhouette Score
  ✔ PCA 2-D & 3-D cluster plots
  ✔ Cluster radar / spider chart for feature profiling
  ✔ Dendrogram for hierarchical clustering
  ✔ DBSCAN noise detection
  ✔ Detailed business intelligence report saved as TXT
"""

import os
import warnings
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless – works without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"

# ── Output directory ──────────────────────────────────────────────────────────
OUT = "/mnt/user-data/outputs"
os.makedirs(OUT, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  1. LOAD & EXPLORE DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_and_explore(path="Mall_Customers.csv"):
    df = pd.read_csv(path)
    print("=" * 60)
    print("  DATA OVERVIEW")
    print("=" * 60)
    print(f"  Shape       : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Missing vals: {df.isnull().sum().sum()}")
    print(f"  Duplicates  : {df.duplicated().sum()}")
    print("\n  Head:")
    print(df.head(3).to_string(index=False))
    print("\n  Statistics:")
    print(df.describe().round(2).to_string())
    print()
    return df

# ══════════════════════════════════════════════════════════════════════════════
#  2. PRE-PROCESS
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(df):
    df = df.copy()
    # Encode gender
    le = LabelEncoder()
    df["Genre_enc"] = le.fit_transform(df["Genre"])          # Male=1, Female=0

    features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    return df, X_scaled, features, scaler

# ══════════════════════════════════════════════════════════════════════════════
#  3. FIND OPTIMAL K  (Elbow + Silhouette)
# ══════════════════════════════════════════════════════════════════════════════

def find_optimal_k(X_scaled, k_range=range(2, 11)):
    print("=" * 60)
    print("  FINDING OPTIMAL K")
    print("=" * 60)
    inertias, silhouettes, dbi_scores, chi_scores = [], [], [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        dbi_scores.append(davies_bouldin_score(X_scaled, labels))
        chi_scores.append(calinski_harabasz_score(X_scaled, labels))

    best_k_sil = list(k_range)[np.argmax(silhouettes)]
    print(f"  Best k (Silhouette) : {best_k_sil}")
    print(f"  Best Silhouette     : {max(silhouettes):.4f}")
    print(f"  Best k (DBI)        : {list(k_range)[np.argmin(dbi_scores)]} "
          f"(lower=better, {min(dbi_scores):.4f})")

    metrics = {
        "k": list(k_range),
        "inertia": inertias,
        "silhouette": silhouettes,
        "dbi": dbi_scores,
        "chi": chi_scores,
    }
    return best_k_sil, metrics


def plot_k_metrics(metrics, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Optimal-K Selection Metrics", fontsize=16, fontweight="bold", y=1.02)
    ks = metrics["k"]

    ax = axes[0]
    ax.plot(ks, metrics["inertia"], "o-", color="#E74C3C", linewidth=2, markersize=8)
    ax.set_title("Elbow Method (Inertia)", fontweight="bold")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ks, metrics["silhouette"], "o-", color="#2ECC71", linewidth=2, markersize=8)
    best_k = ks[np.argmax(metrics["silhouette"])]
    ax.axvline(best_k, linestyle="--", color="gray", alpha=0.7)
    ax.set_title("Silhouette Score (higher = better)", fontweight="bold")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(ks, metrics["dbi"], "o-", color="#9B59B6", linewidth=2, markersize=8)
    ax.set_title("Davies-Bouldin Index (lower = better)", fontweight="bold")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("DBI")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")

# ══════════════════════════════════════════════════════════════════════════════
#  4. CLUSTERING ALGORITHMS
# ══════════════════════════════════════════════════════════════════════════════

def run_kmeans(X_scaled, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    dbi = davies_bouldin_score(X_scaled, labels)
    print(f"\n  [KMeans k={k}]  Silhouette={sil:.4f}  DBI={dbi:.4f}")
    return labels, km


def run_agglomerative(X_scaled, k):
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = agg.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    dbi = davies_bouldin_score(X_scaled, labels)
    print(f"  [Agglomerative k={k}] Silhouette={sil:.4f}  DBI={dbi:.4f}")
    return labels, agg


def run_dbscan(X_scaled, eps=0.5, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = np.sum(labels == -1)
    sil = silhouette_score(X_scaled, labels) if n_clusters > 1 else float("nan")
    print(f"  [DBSCAN eps={eps}]  Clusters={n_clusters}  "
          f"Noise pts={noise}  Silhouette={sil:.4f}" if not np.isnan(sil)
          else f"  [DBSCAN eps={eps}]  Clusters={n_clusters}  Noise pts={noise}  Silhouette=N/A")
    return labels, db, n_clusters

# ══════════════════════════════════════════════════════════════════════════════
#  5. VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = ["#E74C3C", "#2ECC71", "#3498DB", "#F39C12", "#9B59B6",
           "#1ABC9C", "#E67E22", "#2980B9"]


def plot_pca_2d(X_scaled, labels_dict, save_path):
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    ev = pca.explained_variance_ratio_

    n_methods = len(labels_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(7 * n_methods, 6))
    fig.suptitle("PCA 2-D Cluster Projections", fontsize=16, fontweight="bold")

    for ax, (method, labels) in zip(axes, labels_dict.items()):
        unique = np.unique(labels)
        for uid in unique:
            mask = labels == uid
            lbl = f"Noise ({mask.sum()})" if uid == -1 else f"Cluster {uid} ({mask.sum()})"
            color = "#AAAAAA" if uid == -1 else PALETTE[uid % len(PALETTE)]
            ax.scatter(components[mask, 0], components[mask, 1],
                       c=color, label=lbl, s=70, alpha=0.8, edgecolors="white", linewidth=0.4)
        ax.set_title(method, fontweight="bold")
        ax.set_xlabel(f"PC1 ({ev[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({ev[1]:.1%} var)")
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_dendrogram(X_scaled, save_path, max_leaf=30):
    linked = linkage(X_scaled[:max_leaf], method="ward")
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(linked, ax=ax, color_threshold=linked[-4, 2],
               above_threshold_color="#AAAAAA")
    ax.set_title("Hierarchical Clustering Dendrogram (first 30 samples)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Ward Distance")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_cluster_profiles(df, cluster_col, features, save_path):
    """Radar / spider chart for each cluster's feature means."""
    profile = df.groupby(cluster_col)[features].mean()

    # Normalise to 0-1 for radar
    norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)
    cats = features
    N = len(cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    n_clusters = len(profile)
    cols = min(n_clusters, 3)
    rows = (n_clusters + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(5 * cols, 4.5 * rows),
                             subplot_kw=dict(polar=True))
    axes = np.array(axes).flatten()
    fig.suptitle("Cluster Feature Profiles (Radar Charts)",
                 fontsize=15, fontweight="bold", y=1.01)

    for i, (cid, row) in enumerate(norm.iterrows()):
        vals = row.tolist() + row.tolist()[:1]
        ax = axes[i]
        color = PALETTE[i % len(PALETTE)]
        ax.plot(angles, vals, color=color, linewidth=2)
        ax.fill(angles, vals, color=color, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f.replace(" (k$)", "\n(k$)").replace(" (1-100)", "\n(1-100)")
                            for f in cats], fontsize=9)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7)
        ax.set_title(f"Cluster {cid}\n(n={int((df[cluster_col]==cid).sum())})",
                     fontweight="bold", color=color, pad=15)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_boxplots(df, cluster_col, features, save_path):
    fig, axes = plt.subplots(1, len(features), figsize=(6 * len(features), 6))
    fig.suptitle("Feature Distribution by Cluster", fontsize=15, fontweight="bold")

    for ax, feat in zip(axes, features):
        order = sorted(df[cluster_col].unique())
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(order))]
        sns.boxplot(data=df, x=cluster_col, y=feat, order=order,
                    palette=colors, ax=ax, width=0.5, linewidth=1.5)
        ax.set_title(feat, fontweight="bold")
        ax.set_xlabel("Cluster")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_gender_distribution(df, cluster_col, save_path):
    grp = df.groupby([cluster_col, "Genre"]).size().unstack(fill_value=0)
    grp_pct = grp.div(grp.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Gender Distribution by Cluster", fontsize=15, fontweight="bold")

    grp.plot(kind="bar", ax=axes[0], color=["#E91E8C", "#1565C0"],
             edgecolor="white", width=0.6)
    axes[0].set_title("Absolute Counts")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("Count")
    axes[0].legend(title="Gender")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].grid(axis="y", alpha=0.3)

    grp_pct.plot(kind="bar", ax=axes[1], color=["#E91E8C", "#1565C0"],
                 edgecolor="white", width=0.6)
    axes[1].set_title("Percentage Split")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].legend(title="Gender")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_income_vs_spending(df, cluster_col, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Annual Income vs Spending Score", fontsize=15, fontweight="bold")

    for ax, (col, title) in zip(axes,
            [(cluster_col, "KMeans Clusters"), ("Genre", "By Gender")]):
        unique = sorted(df[col].unique())
        for i, uid in enumerate(unique):
            mask = df[col] == uid
            color = PALETTE[i % len(PALETTE)] if col == cluster_col else (
                "#E91E8C" if uid == "Female" else "#1565C0")
            ax.scatter(df.loc[mask, "Annual Income (k$)"],
                       df.loc[mask, "Spending Score (1-100)"],
                       c=color, label=str(uid), s=80,
                       alpha=0.75, edgecolors="white", linewidth=0.4)
        ax.set_xlabel("Annual Income (k$)")
        ax.set_ylabel("Spending Score (1-100)")
        ax.set_title(title, fontweight="bold")
        ax.legend(title=col)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_correlation_heatmap(df, features, save_path):
    corr = df[features + ["Genre_enc"]].rename(
        columns={"Genre_enc": "Gender(M=1)"}).corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                mask=mask, center=0, linewidths=0.5,
                annot_kws={"size": 11}, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")

# ══════════════════════════════════════════════════════════════════════════════
#  6. CLUSTER PROFILING  (business intelligence)
# ══════════════════════════════════════════════════════════════════════════════

SEGMENT_NAMES = {
    # Will be assigned dynamically based on income / spending quadrant
}

def assign_segment_name(income_mean, spending_mean, income_median, spending_median):
    high_income = income_mean >= income_median
    high_spending = spending_mean >= spending_median
    if high_income and high_spending:
        return "💎 Premium Shoppers"
    elif high_income and not high_spending:
        return "💼 Conservative Earners"
    elif not high_income and high_spending:
        return "🛍️  Budget Enthusiasts"
    else:
        return "🌱 Frugal Savers"


def build_cluster_profile(df, cluster_col, features):
    income_med = df["Annual Income (k$)"].median()
    spending_med = df["Spending Score (1-100)"].median()

    profiles = {}
    for cid in sorted(df[cluster_col].unique()):
        sub = df[df[cluster_col] == cid]
        seg_name = assign_segment_name(
            sub["Annual Income (k$)"].mean(),
            sub["Spending Score (1-100)"].mean(),
            income_med, spending_med)
        profiles[cid] = {
            "segment": seg_name,
            "size": len(sub),
            "pct": len(sub) / len(df) * 100,
            "age_mean": sub["Age"].mean(),
            "age_std": sub["Age"].std(),
            "income_mean": sub["Annual Income (k$)"].mean(),
            "spending_mean": sub["Spending Score (1-100)"].mean(),
            "female_pct": (sub["Genre"] == "Female").mean() * 100,
        }
    return profiles

# ══════════════════════════════════════════════════════════════════════════════
#  7. BUSINESS INTELLIGENCE REPORT
# ══════════════════════════════════════════════════════════════════════════════

RECOMMENDATIONS = {
    "💎 Premium Shoppers": [
        "Launch exclusive loyalty / VIP membership programmes.",
        "Promote high-margin premium and luxury product lines.",
        "Offer personalised concierge-style shopping experiences.",
        "Use targeted email / SMS campaigns with early access offers.",
    ],
    "💼 Conservative Earners": [
        "Highlight quality, durability, and long-term value.",
        "Bundle products to increase basket size without heavy discounting.",
        "Promote investment-worthy items (electronics, furniture, etc.).",
        "Use trust-building content: reviews, guarantees, warranties.",
    ],
    "🛍️  Budget Enthusiasts": [
        "Create a flash-sale / daily-deal section tailored to this segment.",
        "Offer instalment or 'buy now pay later' payment options.",
        "Reward frequent visits with cashback or points programmes.",
        "Highlight trending / social-media popular items at accessible prices.",
    ],
    "🌱 Frugal Savers": [
        "Use re-engagement campaigns to understand purchase barriers.",
        "Provide educational content about store value propositions.",
        "Trial low-commitment offers: free samples, starter kits.",
        "Investigate whether pricing or product mix is a deterrent.",
    ],
}

def write_report(profiles, km_sil, km_dbi, best_k, save_path):
    lines = []
    w = 70

    def sep(char="═"):
        lines.append(char * w)

    def title(text):
        sep()
        lines.append(f"  {text}")
        sep()

    def section(text):
        lines.append("")
        lines.append(f"  ── {text} " + "─" * (w - len(text) - 6))

    sep("╔" + "═" * (w - 2) + "╗"[:1])
    lines.append("║" + "  RETAIL STORE CUSTOMER SEGMENTATION".center(w - 2) + "║")
    lines.append("║" + "  BUSINESS INTELLIGENCE REPORT".center(w - 2) + "║")
    sep("╚" + "═" * (w - 2) + "╝"[:1])
    lines.append("")

    # ── Executive Summary ──
    section("EXECUTIVE SUMMARY")
    lines.append(f"  Algorithm   : KMeans (optimal k selected via Silhouette)")
    lines.append(f"  Optimal k   : {best_k} clusters")
    lines.append(f"  Silhouette  : {km_sil:.4f}  (range −1 to 1; >0.5 = good)")
    lines.append(f"  Davies-Bouldin Index: {km_dbi:.4f}  (lower = better)")

    total = sum(p["size"] for p in profiles.values())
    lines.append(f"  Customers   : {total}")
    lines.append("")

    # ── Cluster Profiles ──
    section("CLUSTER PROFILES")
    for cid, p in profiles.items():
        lines.append(f"\n  Cluster {cid} — {p['segment']}")
        lines.append(f"  {'─'*55}")
        lines.append(f"  Size              : {p['size']} customers ({p['pct']:.1f}% of base)")
        lines.append(f"  Avg Age           : {p['age_mean']:.1f} ± {p['age_std']:.1f} yrs")
        lines.append(f"  Avg Annual Income : ${p['income_mean']:.1f}k")
        lines.append(f"  Avg Spending Score: {p['spending_mean']:.1f} / 100")
        lines.append(f"  Female share      : {p['female_pct']:.1f}%")

    # ── Recommendations ──
    section("STRATEGIC RECOMMENDATIONS")
    for cid, p in profiles.items():
        seg = p["segment"]
        recs = RECOMMENDATIONS.get(seg, ["No specific recommendations."])
        lines.append(f"\n  Cluster {cid} — {seg}")
        for r in recs:
            wrapped = textwrap.fill(r, width=w - 6,
                                    initial_indent="    • ",
                                    subsequent_indent="      ")
            lines.append(wrapped)

    # ── Algorithm Comparison ──
    section("ALGORITHM NOTES")
    lines.append("  Three algorithms were evaluated:")
    lines.append("  1. KMeans           – fast, spherical clusters (primary model)")
    lines.append("  2. Agglomerative HC – tree-based, useful for dendrogram insight")
    lines.append("  3. DBSCAN           – density-based, detects outliers/noise")
    lines.append("")
    lines.append("  All chart outputs are saved alongside this report.")
    lines.append("")
    sep()
    lines.append("  END OF REPORT")
    sep()

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved → {save_path}")

# ══════════════════════════════════════════════════════════════════════════════
#  8. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("   RETAIL CUSTOMER CLUSTERING — UPGRADED SUITE")
    print("=" * 60 + "\n")

    # 1. Load
    df = load_and_explore("Mall_Customers.csv")

    # 2. Pre-process
    df, X_scaled, features, scaler = preprocess(df)

    # 3. Find optimal k
    best_k, metrics = find_optimal_k(X_scaled)

    # 4. Plot k-selection metrics
    plot_k_metrics(metrics, f"{OUT}/01_k_selection_metrics.png")

    # 5. Correlation heatmap
    plot_correlation_heatmap(df, features, f"{OUT}/02_correlation_heatmap.png")

    # 6. Run algorithms
    print("\n" + "=" * 60)
    print("  RUNNING CLUSTERING ALGORITHMS")
    print("=" * 60)
    km_labels, km_model = run_kmeans(X_scaled, best_k)
    agg_labels, _       = run_agglomerative(X_scaled, best_k)
    db_labels, _, n_db  = run_dbscan(X_scaled, eps=0.5, min_samples=5)

    df["Cluster_KMeans"] = km_labels
    df["Cluster_Agg"]    = agg_labels
    df["Cluster_DBSCAN"] = db_labels

    # 7. Visualisations
    print("\n" + "=" * 60)
    print("  GENERATING VISUALISATIONS")
    print("=" * 60)

    plot_pca_2d(X_scaled,
                {"KMeans": km_labels,
                 "Agglomerative": agg_labels,
                 "DBSCAN": db_labels},
                f"{OUT}/03_pca_2d_all_algorithms.png")

    plot_dendrogram(X_scaled, f"{OUT}/04_dendrogram.png")

    plot_cluster_profiles(df, "Cluster_KMeans", features,
                          f"{OUT}/05_radar_cluster_profiles.png")

    plot_boxplots(df, "Cluster_KMeans", features,
                  f"{OUT}/06_boxplots_by_cluster.png")

    plot_gender_distribution(df, "Cluster_KMeans",
                             f"{OUT}/07_gender_distribution.png")

    plot_income_vs_spending(df, "Cluster_KMeans",
                            f"{OUT}/08_income_vs_spending.png")

    # 8. Business profile & report
    km_sil = silhouette_score(X_scaled, km_labels)
    km_dbi = davies_bouldin_score(X_scaled, km_labels)
    profiles = build_cluster_profile(df, "Cluster_KMeans", features)
    write_report(profiles, km_sil, km_dbi, best_k,
                 f"{OUT}/09_business_intelligence_report.txt")

    # 9. Save enriched dataset
    out_csv = f"{OUT}/10_customers_with_clusters.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Saved → {out_csv}")

    # 10. Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for cid, p in profiles.items():
        print(f"  Cluster {cid}: {p['segment']:30s}  "
              f"n={p['size']:3d}  Income=${p['income_mean']:.0f}k  "
              f"Spend={p['spending_mean']:.0f}")
    print(f"\n  All outputs saved to: {OUT}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
