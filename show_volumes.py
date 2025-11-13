# kidney_volume_visualization.py

import matplotlib.pyplot as plt

# Data (in cm³)
volumes = {
    "Kidney": 139.82,
    "Cysts": 8.4619,
    "Tumors": 0.13
}

labels = list(volumes.keys())
values = list(volumes.values())

# Colors (Kidney, Cysts, Tumors)
colors = ["#76508A", "#65A9A2", "#51538B"]

# Compute total and percentages
total_volume = sum(values)
percentages = {k: (v / total_volume) * 100 for k, v in volumes.items()}


def autopct_format(pct):
    """Hide labels for very small slices (< 0.5%)."""
    return "" if pct < 0.5 else f"{pct:.2f}%"


# --- PIE CHART ---
plt.figure(figsize=(8, 8))

wedges, texts, autotexts = plt.pie(
    values,
    labels=None,
    autopct=autopct_format,
    startangle=90,
    pctdistance=0.75,
    colors=colors
)

# Tweak text appearance
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_color("white")

plt.title("Kidney Composition by Volume (cm³)", fontsize=14)
plt.axis("equal")  # Circular pie chart

# Legend on the side
plt.legend(
    wedges,
    labels,
    title="Components",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    fontsize=10,
    title_fontsize=11
)

plt.tight_layout()
plt.show()


# --- BAR CHART ---
plt.figure(figsize=(8, 6))
bars = plt.bar(labels, values, color=colors)

plt.title("Absolute Volumes of Kidney, Cysts, and Tumors", fontsize=14)
plt.ylabel("Volume (cm³)", fontsize=12)

# Add value labels
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 0.5,
        f"{yval:.4f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()
plt.show()


# --- PRINT SUMMARY ---
print("Volume Summary (cm³):")
for k, v in volumes.items():
    print(f"  {k}: {v:.4f} cm³ ({percentages[k]:.2f}%)")
print(f"  Total: {total_volume:.4f} cm³")
