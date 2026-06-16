import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

AXIS_LABEL_SIZE = 13
TICK_LABEL_SIZE = 12
LEGEND_SIZE = 11
ANNOTATION_SIZE = 12
FORMULA_SIZE = 11
CAPTION_SIZE = 13

# ============================================================
# Figure: Theoretical Analysis of IKD
# Theme:
# (a) Probability-aware weighting
# (b) Fisher geometry vs Euclidean geometry
# (c) CE-KL equivalence
# (d) Identical CE/KL trajectories vs MSE deviation
# ============================================================

np.random.seed(0)

fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))


def add_bottom_caption(ax, caption):
    ax.text(
        0.5,
        -0.38,
        caption,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=CAPTION_SIZE,
        linespacing=1.15
    )

# ============================================================
# (a) Probability-aware weighting
# ============================================================

ax = axes[0]

classes = np.arange(6)

# Example benign prediction distribution p
p = np.array([0.74, 0.10, 0.06, 0.04, 0.035, 0.025])

# Probability-aware weighting in the Fisher metric
weights = 1.0 / p
weights = weights / weights.max()

ax.bar(
    classes - 0.18,
    p,
    width=0.35,
    label=r"$p_i$"
)

ax.bar(
    classes + 0.18,
    weights,
    width=0.35,
    label=r"normalized $1/p_i$"
)

ax.set_xticks(classes)
ax.set_xticklabels([f"C{i}" for i in classes])
ax.set_ylabel("Magnitude", fontsize=AXIS_LABEL_SIZE)
add_bottom_caption(ax, "(a) Probability-aware\nweighting")

ax.text(
    0.5,
    1.06,
    r"CE/KL local metric: $J^\top D^{-1}J$" "\n"
    r"$D=\mathrm{diag}(p)$",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=FORMULA_SIZE,
    linespacing=1.15,
    bbox=dict(boxstyle="round", alpha=0.15)
)

ax.legend(frameon=False, fontsize=LEGEND_SIZE)


# ============================================================
# (b) Fisher geometry vs Euclidean geometry
# ============================================================

ax = axes[1]

theta = np.linspace(0, 2 * np.pi, 300)

# Fisher geometry: anisotropic ellipse
x1 = 1.75 * np.cos(theta)
y1 = 0.65 * np.sin(theta)

angle = np.deg2rad(35)
rotation = np.array(
    [
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ]
)

fisher_ellipse = rotation @ np.vstack([x1, y1])

# MSE geometry: isotropic Euclidean circle
x2 = 1.10 * np.cos(theta)
y2 = 1.10 * np.sin(theta)

ax.plot(
    fisher_ellipse[0],
    fisher_ellipse[1],
    label=r"CE/KL Fisher metric"
)

ax.plot(
    x2,
    y2,
    linestyle="--",
    label=r"MSE Euclidean metric"
)

ax.axhline(0, linewidth=0.8)
ax.axvline(0, linewidth=0.8)

# SDS direction arrow
arrow = FancyArrowPatch(
    (-0.1, -0.05),
    (1.55 * np.cos(angle), 1.55 * np.sin(angle)),
    arrowstyle="->",
    mutation_scale=14,
    linewidth=1.5
)

ax.add_patch(arrow)

ax.text(
    0.96,
    0.89,
    "SDS direction",
    transform=ax.transAxes,
    ha="right",
    va="center",
    fontsize=ANNOTATION_SIZE
)

ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-1.8, 1.8)
ax.set_xlabel("input direction 1", fontsize=AXIS_LABEL_SIZE)
ax.set_ylabel("input direction 2", fontsize=AXIS_LABEL_SIZE)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.36),
    frameon=False,
    fontsize=LEGEND_SIZE
)
add_bottom_caption(ax, "(b) Fisher geometry vs\nEuclidean geometry")


# ============================================================
# (c) CE-KL equivalence under fixed anchor
# ============================================================

ax = axes[2]

t = np.linspace(0, 1, 100)

# Toy KL curve
kl_curve = 0.15 + 0.85 * (t - 0.15) ** 2 + 0.2 * t

# Constant entropy offset H(p)
entropy_const = 0.45
ce_curve = kl_curve + entropy_const

ax.plot(
    t,
    kl_curve,
    label=r"$KL(p\|q)$"
)

ax.plot(
    t,
    ce_curve,
    linestyle="--",
    label=r"$CE(p,q)=KL(p\|q)+H(p)$"
)

# Show constant vertical gap H(p)
x_gap = 0.72
kl_y = 0.15 + 0.85 * (x_gap - 0.15) ** 2 + 0.2 * x_gap
ce_y = kl_y + entropy_const

ax.annotate(
    "",
    xy=(x_gap, ce_y),
    xytext=(x_gap, kl_y),
    arrowprops=dict(arrowstyle="<->", linewidth=1.2)
)

ax.text(
    x_gap + 0.03,
    (kl_y + ce_y) / 2,
    r"$H(p)$",
    va="center",
    fontsize=ANNOTATION_SIZE
)

ax.text(
    0.03,
    0.95,
    r"$p=f_\phi(x)$ is fixed" "\n"
    r"$\nabla_{x^{adv}}CE=\nabla_{x^{adv}}KL$" "\n"
    r"$\nabla^2_{x^{adv}}CE=\nabla^2_{x^{adv}}KL$",
    transform=ax.transAxes,
    va="top",
    fontsize=FORMULA_SIZE,
    bbox=dict(boxstyle="round", alpha=0.15)
)

ax.set_xlabel("optimization state", fontsize=AXIS_LABEL_SIZE)
ax.set_ylabel("objective value", fontsize=AXIS_LABEL_SIZE)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.36),
    frameon=False,
    fontsize=LEGEND_SIZE
)
add_bottom_caption(ax, "(c) CE-KL equivalence\nunder fixed anchor")


# ============================================================
# (d) Identical CE/KL trajectories vs MSE deviation
# ============================================================

ax = axes[3]

steps = np.arange(10)

# CE and KL share exactly the same trajectory
x_ce = np.linspace(0, 1.6, len(steps))
y_ce = (
    0.25 * np.sin(np.linspace(0, 2.6, len(steps)))
    + np.linspace(0, 1.15, len(steps))
)

# MSE deviates from the CE/KL trajectory
x_mse = x_ce + 0.12 * steps / steps.max()
y_mse = (
    y_ce
    - 0.05 * steps
    - 0.15 * np.sin(np.linspace(0, 1.8, len(steps)))
)

ax.plot(
    x_ce,
    y_ce,
    marker="o",
    label="CE trajectory"
)

ax.plot(
    x_ce,
    y_ce,
    linestyle="--",
    marker="x",
    label="KL trajectory"
)

ax.plot(
    x_mse,
    y_mse,
    marker="s",
    label="MSE trajectory"
)

ax.annotate(
    "start",
    xy=(x_ce[0], y_ce[0]),
    xytext=(x_ce[0] - 0.15, y_ce[0] + 0.15),
    fontsize=ANNOTATION_SIZE,
    arrowprops=dict(arrowstyle="->", linewidth=1.0)
)

ax.annotate(
    "transferable direction",
    xy=(x_ce[-1], y_ce[-1]),
    xytext=(x_ce[-1] - 0.95, y_ce[-1] + 0.35),
    fontsize=ANNOTATION_SIZE,
    arrowprops=dict(arrowstyle="->", linewidth=1.0)
)

ax.annotate(
    "MSE deviation",
    xy=(x_mse[-1], y_mse[-1]),
    xytext=(x_mse[-1] - 0.70, y_mse[-1] - 0.45),
    fontsize=ANNOTATION_SIZE,
    arrowprops=dict(arrowstyle="->", linewidth=1.0)
)

ax.set_xlabel("input-space direction 1", fontsize=AXIS_LABEL_SIZE)
ax.set_ylabel("input-space direction 2", fontsize=AXIS_LABEL_SIZE)
ax.legend(loc="upper left", frameon=False, fontsize=LEGEND_SIZE)
add_bottom_caption(ax, "(d) Identical CE/KL\ntrajectories")


# ============================================================
# Save figure
# ============================================================

fig.subplots_adjust(
    left=0.04,
    right=0.99,
    top=0.80,
    bottom=0.38,
    wspace=0.42
)

for ax in axes:
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.xaxis.label.set_fontsize(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_fontsize(AXIS_LABEL_SIZE)

fig.savefig(
    "revised_section_III_C_proof_figure.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    "revised_section_III_C_proof_figure.pdf",
    bbox_inches="tight"
)

plt.show()
