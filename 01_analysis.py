"""
Bangladesh wildlife extinction-risk analysis (2000 / 2015 baselines -> 2030 projection)

DATA POLICY
- Every number in VERIFIED_* comes from a published source (see data/sources.csv).
- Anything marked ASSUMPTION is a modelling choice, not an observation.
- No yearly 2010-2026 species series exists publicly. The next Bangladesh Red List
  (work began 2026) will be the first real test of this projection.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/mnt/user-data/outputs/bd_wildlife"
D, P = f"{BASE}/data", f"{BASE}/plots"
rng = np.random.default_rng(2030)
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})

# ------------------------------------------------------------------ VERIFIED DATA
groups = ["Mammals", "Birds", "Reptiles", "Amphibians", "Freshwater fish", "Crustaceans", "Butterflies"]
assessed = [138, 566, 167, 49, 253, 141, 305]            # IUCN Bangladesh 2015 (sum 1619)
threatened = [38, 39, 38, 10, 64, 13, 188]               # CR+EN+VU (sum 390)
extinct_2015 = [11, 19, 1, 0, 0, 0, 0]                   # regionally extinct (sum 31)
extinct_2000 = [10, 2, 1, 0, 0, 0, 0]                    # 2000 Red List (sum 13)
rli_2015 = [0.55, 0.91, 0.76, 0.85, 0.81, 0.90, 0.64]    # IUCN Bangladesh blog 2017
# exact CR/EN/VU only published for these two groups:
exact = {"Mammals": (17, 12, 9), "Amphibians": (2, 3, 5)}
NAT = {"CR": 56, "EN": 181, "VU": 153, "NT": 90, "LC": 802, "RE": 31}  # national totals 2015

assert sum(assessed) == 1619 and sum(threatened) == 390
assert sum(extinct_2015) == 31 and sum(extinct_2000) == 13
assert NAT["CR"] + NAT["EN"] + NAT["VU"] == 390

# ASSUMPTION A1: split of CR/EN/VU for groups without published detail =
# national remainder, distributed proportionally to each group's threatened count.
rem = np.array([NAT["CR"] - 17 - 2, NAT["EN"] - 12 - 3, NAT["VU"] - 9 - 5], float)  # 37,166,139
rem_groups = [g for g in groups if g not in exact]
rem_thr = np.array([threatened[groups.index(g)] for g in rem_groups], float)
assert abs(rem_thr.sum() - rem.sum()) < 1e-9
cat = {}
for g in groups:
    if g in exact:
        cat[g] = np.array(exact[g], float)
    else:
        cat[g] = rem * (threatened[groups.index(g)] / rem_thr.sum())

tab = pd.DataFrame({
    "group": groups, "assessed_2015": assessed, "threatened_2015": threatened,
    "regionally_extinct_2000": extinct_2000, "regionally_extinct_2015": extinct_2015,
    "RLI_2015": rli_2015,
    "CR": [cat[g][0] for g in groups], "EN": [cat[g][1] for g in groups], "VU": [cat[g][2] for g in groups],
    "CR_EN_VU_source": ["published" if g in exact else "ASSUMPTION A1 (proportional)" for g in groups],
})
tab["pct_threatened"] = (100 * tab.threatened_2015 / tab.assessed_2015).round(1)
tab["pct_extinct"] = (100 * tab.regionally_extinct_2015 / (tab.assessed_2015)).round(2)
tab.to_csv(f"{D}/group_summary_2000_2015.csv", index=False)

# ------------------------------------------------------------------ MONTE CARLO
# IUCN Criterion E thresholds (upper-bound reference probabilities):
#   CR >=50% in 10 y ; EN >=20% in 20 y ; VU >=10% in 100 y
lam = {"CR": -np.log(0.5) / 10, "EN": -np.log(0.8) / 20, "VU": -np.log(0.9) / 100}
# ASSUMPTION A2: realised fraction r of threshold hazard ~ Beta(2,5) (mean 0.29),
# because real-world extinction is far slower than the criterion thresholds.
# ASSUMPTION A3: independent species, constant hazard, same r shared across groups (correlated scenario).
years = np.arange(2015, 2031)
NSIM = 20000
r = rng.beta(2, 5, NSIM)

def simulate(group_cat, r_draws):
    out = np.zeros((len(r_draws), len(years)))
    for ti, y in enumerate(years):
        t = y - 2015
        tot = np.zeros(len(r_draws))
        for ci, c in enumerate(["CR", "EN", "VU"]):
            n = int(round(group_cat[ci]))
            p = 1 - np.exp(-lam[c] * r_draws * t)
            tot += rng.binomial(n, p)
        out[:, ti] = tot
    return out

res, rows = {}, []
for g in groups:
    sim = simulate(cat[g], r)
    res[g] = sim
    for y in (2026, 2030):
        k = list(years).index(y)
        rows.append([g, y, sim[:, k].mean(), *np.percentile(sim[:, k], [5, 95])])
tot_sim = sum(res.values())
for y in (2026, 2030):
    k = list(years).index(y)
    rows.append(["ALL GROUPS", y, tot_sim[:, k].mean(), *np.percentile(tot_sim[:, k], [5, 95])])
proj = pd.DataFrame(rows, columns=["group", "year", "expected_new_regional_extinctions_since_2015", "p05", "p95"]).round(2)
proj.to_csv(f"{D}/projection_new_extinctions.csv", index=False)

# sensitivity: fixed r grid
grid = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
sens = []
for rr in grid:
    e = 0
    for g in groups:
        for ci, c in enumerate(["CR", "EN", "VU"]):
            e += cat[g][ci] * (1 - np.exp(-lam[c] * rr * 15))
    sens.append(e)
pd.DataFrame({"realised_fraction_r": grid, "expected_extinctions_2015_to_2030": np.round(sens, 1)}).to_csv(
    f"{D}/sensitivity_r.csv", index=False)

# RLI lower-bound projection: only extinctions counted (CR->EX raises score by 1 per species)
rli_rows = []
for gi, g in enumerate(groups):
    k = list(years).index(2030)
    d = res[g][:, k] / (5 * assessed[gi])
    rli_rows.append([g, rli_2015[gi], rli_2015[gi] - d.mean(),
                     rli_2015[gi] - np.percentile(d, 95), rli_2015[gi] - np.percentile(d, 5)])
rli = pd.DataFrame(rli_rows, columns=["group", "RLI_2015", "RLI_2030_mean", "RLI_2030_low", "RLI_2030_high"]).round(4)
rli.to_csv(f"{D}/rli_projection_2030.csv", index=False)

# ------------------------------------------------------------------ FOREST (verified anchors only)
forest = pd.DataFrame({
    "year": [2001, 2010, 2011, 2012, 2017, 2022, 2023, 2024],
    "tree_cover_loss_ha": [4481, 4540, 4060, 5280, 28328, 13800, 17770, 15911],
    "source": ["GFW via OWID/geofactbook", "GFW via geofactbook (approx, 4.54K)", "GFW via geofactbook (approx, 4.06K)",
               "GFW via geofactbook (approx, 5.28K)", "GFW cited in Observer BD", "Dhaka Tribune citing GFW",
               "geofactbook / Dhaka Tribune", "geofactbook (GFW via OWID)"]})
forest.to_csv(f"{D}/forest_tree_cover_loss_verified_points.csv", index=False)
# indicative linear trend with bootstrap CI (8 points -> treat as indicative only)
x, y = forest.year.values, forest.tree_cover_loss_ha.values
bt = []
for _ in range(4000):
    i = rng.integers(0, len(x), len(x))
    if len(set(x[i])) < 3:
        continue
    bt.append(np.polyval(np.polyfit(x[i], y[i], 1), 2030))
f30 = np.percentile(bt, [5, 50, 95])
pd.DataFrame({"metric": ["indicative_2030_tree_cover_loss_ha_p05", "p50", "p95"], "value": f30.round(0)}).to_csv(
    f"{D}/forest_indicative_2030.csv", index=False)

# ------------------------------------------------------------------ PLOTS
C = "#2b6a4f"; R = "#b5372f"; O = "#d98b2b"; G = "#8a8a8a"

# 1 assessed vs threatened
fig, ax = plt.subplots(figsize=(7, 4))
xx = np.arange(len(groups)); w = .4
ax.bar(xx - w/2, assessed, w, color=G, label="Assessed")
ax.bar(xx + w/2, threatened, w, color=R, label="Threatened (CR+EN+VU)")
ax.set_xticks(xx); ax.set_xticklabels(groups, rotation=30, ha="right"); ax.set_ylabel("Species")
ax.set_title("Bangladesh Red List 2015: species assessed vs threatened"); ax.legend()
fig.tight_layout(); fig.savefig(f"{P}/01_assessed_vs_threatened.png"); plt.close()

# 2 % threatened
fig, ax = plt.subplots(figsize=(7, 4))
o = tab.sort_values("pct_threatened")
ax.barh(o.group, o.pct_threatened, color=R)
for i, v in enumerate(o.pct_threatened): ax.text(v + .8, i, f"{v}%", va="center")
ax.set_xlabel("% of assessed species threatened"); ax.set_title("Share threatened by group (2015)")
fig.tight_layout(); fig.savefig(f"{P}/02_pct_threatened.png"); plt.close()

# 3 extinct 2000 vs 2015
fig, ax = plt.subplots(figsize=(6, 4))
tx = ["Mammals", "Birds", "Reptiles"]
a = [10, 2, 1]; b = [11, 19, 1]
xx = np.arange(3)
ax.bar(xx - .2, a, .4, color=O, label="2000 Red List (13)"); ax.bar(xx + .2, b, .4, color=R, label="2015 Red List (31)")
for i in range(3):
    ax.text(i - .2, a[i] + .3, a[i], ha="center"); ax.text(i + .2, b[i] + .3, b[i], ha="center")
ax.set_xticks(xx); ax.set_xticklabels(tx); ax.set_ylabel("Regionally extinct species")
ax.set_title("Regionally extinct: 2000 vs 2015\n(2015 includes re-assessment, not only new losses)"); ax.legend()
fig.tight_layout(); fig.savefig(f"{P}/03_extinct_2000_vs_2015.png"); plt.close()

# 4 national status composition
fig, ax = plt.subplots(figsize=(6.5, 4))
lab = ["RE", "CR", "EN", "VU", "NT", "LC"]; val = [NAT[k] for k in lab]
col = ["#222", "#b5372f", "#e0652b", "#d9a62b", "#9bb04a", "#4c9a6a"]
ax.bar(lab, val, color=col)
for i, v in enumerate(val): ax.text(i, v + 10, v, ha="center")
ax.set_ylabel("Species (of 1,619)"); ax.set_title("National Red List status composition, 2015")
fig.tight_layout(); fig.savefig(f"{P}/04_status_composition.png"); plt.close()

# 5 RLI 2015
fig, ax = plt.subplots(figsize=(7, 4))
o = tab.sort_values("RLI_2015")
ax.barh(o.group, o.RLI_2015, color=[R if v < .7 else (O if v < .85 else C) for v in o.RLI_2015])
ax.set_xlim(0, 1); ax.set_xlabel("Red List Index (1 = all Least Concern, 0 = all extinct)")
ax.set_title("Red List Index 2015 by animal group")
for i, v in enumerate(o.RLI_2015): ax.text(v + .01, i, f"{v:.2f}", va="center")
fig.tight_layout(); fig.savefig(f"{P}/05_rli_2015.png"); plt.close()

# 6 projection trajectory
fig, ax = plt.subplots(figsize=(7, 4.3))
m = tot_sim.mean(0); lo = np.percentile(tot_sim, 5, 0); hi = np.percentile(tot_sim, 95, 0)
ax.fill_between(years, lo, hi, color=R, alpha=.2, label="90% interval")
ax.plot(years, m, color=R, lw=2, label="Expected")
ax.axvline(2026, color=G, ls="--"); ax.text(2026.1, hi.max()*.05, "today", color=G)
ax.set_xlabel("Year"); ax.set_ylabel("Cumulative NEW regional extinctions since 2015")
ax.set_title("Monte Carlo projection (all groups) - scenario, not observation"); ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(f"{P}/06_projection_trajectory.png"); plt.close()

# 7 by group 2030
fig, ax = plt.subplots(figsize=(7, 4))
p30 = proj[(proj.year == 2030) & (proj.group != "ALL GROUPS")].set_index("group").loc[groups]
ax.barh(groups, p30.expected_new_regional_extinctions_since_2015, color=R,
        xerr=[p30.expected_new_regional_extinctions_since_2015 - p30.p05, p30.p95 - p30.expected_new_regional_extinctions_since_2015],
        error_kw=dict(ecolor="#333", capsize=3))
ax.invert_yaxis(); ax.set_xlabel("Expected new regional extinctions by 2030 (90% interval)")
ax.set_title("Where the projected losses fall")
fig.tight_layout(); fig.savefig(f"{P}/07_projection_by_group_2030.png"); plt.close()

# 8 sensitivity
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(grid, sens, marker="o", color=R)
ax.axvspan(.1, .5, color=O, alpha=.15, label="plausible range (A2)")
ax.set_xlabel("Realised fraction r of IUCN Criterion-E hazard"); ax.set_ylabel("Expected extinctions 2015-2030")
ax.set_title("Sensitivity to the key assumption"); ax.legend()
fig.tight_layout(); fig.savefig(f"{P}/08_sensitivity.png"); plt.close()

# 9 RLI 2015 vs 2030 lower bound
fig, ax = plt.subplots(figsize=(7, 4))
xx = np.arange(len(groups)); w = .4
ax.bar(xx - w/2, rli.RLI_2015, w, color=C, label="2015 (published)")
ax.bar(xx + w/2, rli.RLI_2030_mean, w, color=R, label="2030 (extinctions-only lower bound)")
ax.set_ylim(.4, 1); ax.set_xticks(xx); ax.set_xticklabels(groups, rotation=30, ha="right")
ax.set_ylabel("Red List Index"); ax.set_title("RLI: 2015 vs projected 2030\n(category worsening not modelled, so true decline may be larger)")
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(f"{P}/09_rli_2015_vs_2030.png"); plt.close()

# 10 forest
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(forest.year, forest.tree_cover_loss_ha / 1000, color=R, zorder=3, label="Verified data points (GFW)")
xs = np.arange(2001, 2031)
ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs) / 1000, color=G, ls="--", label="Indicative linear trend")
ax.errorbar([2030], [f30[1] / 1000], yerr=[[(f30[1]-f30[0]) / 1000], [(f30[2]-f30[1]) / 1000]], fmt="s", color=O, label="2030 indicative (90% bootstrap)")
ax.set_xlabel("Year"); ax.set_ylabel("Tree cover loss (thousand ha)")
ax.set_title("Bangladesh annual tree cover loss - only verified points\n(not a full series; 2017 spike 28.3k ha)")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{P}/10_forest_tree_cover_loss.png"); plt.close()

print(proj.to_string()); print(rli.to_string()); print("sens", dict(zip(grid, np.round(sens, 1))))
print("forest 2030 p05/50/95", f30.round(0))
