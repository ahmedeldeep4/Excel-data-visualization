import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SRC = "bank_churn.csv" 
OUT = "charts"
import os
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(SRC)
df["churned"] = (df["churn"] == "Yes").astype(int)

sns.set_theme(style="whitegrid")
PAL = {"No": "#2a9d8f", "Yes": "#e76f51"}

def save(name):
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}.png", dpi=150)
    plt.close()


fig, ax = plt.subplots(figsize=(5, 5))
c = df["churn"].value_counts()
ax.pie(c, labels=c.index, autopct="%1.1f%%", colors=[PAL[i] for i in c.index],
       startangle=90, wedgeprops=dict(width=0.45))
ax.set_title("Overall Churn")
save("01_overall_churn")


for i, col in enumerate(["governorate", "card_type", "salary_transfer", "num_products"], start=2):
    r = df.groupby(col)["churned"].mean().mul(100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=r.index.astype(str), y=r.values, color="#264653", ax=ax)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Churn rate (%)"); ax.set_xlabel(col)
    ax.set_title(f"Churn rate by {col}")
    save(f"{i:02d}_churn_by_{col}")


for i, col in [(6, "complaints_last_year"), (7, "tenure_years")]:
    r = df.groupby(col)["churned"].mean().mul(100)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(r.index, r.values, marker="o", color="#e76f51")
    ax.set_ylabel("Churn rate (%)"); ax.set_xlabel(col)
    ax.set_title(f"Churn rate by {col}")
    save(f"{i:02d}_churn_by_{col}")


fig, ax = plt.subplots(figsize=(7, 4.5))
sns.histplot(data=df, x="age", hue="churn", palette=PAL, kde=True, bins=30, ax=ax)
ax.set_title("Age distribution by churn")
save("08_age_distribution")

fig, ax = plt.subplots(figsize=(6, 4.5))
sns.boxplot(data=df, x="churn", y="credit_score", palette=PAL, hue="churn", legend=False, ax=ax)
ax.set_title("Credit score by churn")
save("09_credit_score_boxplot")


fig, ax = plt.subplots(figsize=(6, 4.5))
sns.boxplot(data=df, x="churn", y="app_logins_month", palette=PAL, hue="churn", legend=False, ax=ax)
ax.set_title("Monthly app logins by churn")
save("10_app_logins_boxplot")


fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=df, x="monthly_salary_egp", y="balance_egp", hue="churn",
                palette=PAL, alpha=0.5, s=18, ax=ax)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_title("Monthly salary vs balance (log scale)")
save("11_salary_vs_balance")

num = df.drop(columns=["customer_id"]).select_dtypes("number")
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation matrix")
save("12_correlation_heatmap")

print("Done")
