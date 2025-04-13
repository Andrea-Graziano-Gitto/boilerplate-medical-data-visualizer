import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas.api.types import CategoricalDtype

# 1. Import data
df = pd.read_csv("medical_examination.csv")

# 2. Add 'overweight' column (BMI > 25)
df['overweight'] = ((df['weight'] / ((df['height'] / 100) ** 2)) > 25).astype(int)

# 3. Normalize cholesterol and gluc
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


# 4. Draw Categorical Plot
def draw_cat_plot():
    # 5. Create DataFrame for cat plot
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"]
    )

    # 6. Group and reformat data
    df_cat = df_cat.groupby(["cardio", "variable", "value"]).size().reset_index(name='total')

    # 7. Convert 'variable' to ordered categorical for consistent order
    cat_order = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    df_cat['variable'] = df_cat['variable'].astype(CategoricalDtype(categories=cat_order, ordered=True))

    # 8. Draw the catplot
    g = sns.catplot(
        x="variable",
        y="total",
        hue="value",
        col="cardio",
        kind="bar",
        data=df_cat
    )
    g.set_axis_labels("variable", "total")

    # 9. Save figure
    g.fig.savefig("catplot.png")
    return g.fig


# 10. Draw Heat Map
def draw_heat_map():
    # 11. Clean the data
    df_heat = df[
        (df["ap_lo"] <= df["ap_hi"]) &
        (df["height"] >= df["height"].quantile(0.025)) &
        (df["height"] <= df["height"].quantile(0.975)) &
        (df["weight"] >= df["weight"].quantile(0.025)) &
        (df["weight"] <= df["weight"].quantile(0.975))
    ]

    # 12. Calculate correlation matrix
    corr = df_heat.corr().round(1)

    # 13. Generate mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Set up matplotlib figure and draw heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        cbar_kws={'shrink': 0.5},
        ax=ax
    )

    # 16. Save figure
    fig.savefig("heatmap.png")
    return fig
