"""
Revision statistics and figures for:
  Benchmarking a generative large language model against students on authentic
  mechanical engineering exams from eight universities

Input : score_final.xlsx  (N = 83 exams; columns: University, Course, student mean,
                           student median, GPT-4o score)
Output: printed statistics (Sections 3.1, 3.2, 3.4, Table 4) and Figures 3-7 in figures/

Usage:
    python llm_me_exams.py                      # reads ./score_final.xlsx, writes ./figures/
    python llm_me_exams.py path/to/scores.xlsx  # read a different file

Requirements: Python >= 3.10, numpy, pandas, scipy, matplotlib, seaborn, openpyxl.
Bootstrap results use a fixed random seed (2026).
"""

import sys
from pathlib import Path

# ---- Configuration ----

DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("score_final.xlsx")   # exam-level scores
SAVE_DIR  = Path("figures")   # where Figures 3-7 are written (PNG + PDF)

SAVE_DIR.mkdir(parents=True, exist_ok=True)
if not DATA_PATH.exists():
    sys.exit(f"Data file not found: {DATA_PATH.resolve()}")
print(f"Reading data from: {DATA_PATH.resolve()}")
print(f"Figures will be saved to: {SAVE_DIR.resolve()}")

import numpy as np
import pandas as pd
from scipy import stats

rng = np.random.default_rng(2026)
B = 10000   # bootstrap resamples

df = pd.read_excel(DATA_PATH)
# columns (in order): University, Course, student mean, student median, GPT-4o score
df.columns = ['University', 'Course', 'hm', 'hmed', 'gpt']
df[['University', 'Course']] = df[['University', 'Course']].ffill()   # fill merged-cell blanks
df['course_id'] = df['University'] + ' | ' + df['Course']

hm, hmed, g = df['hm'].values, df['hmed'].values, df['gpt'].values
n = len(df)
print(f"N = {n} exams, {df.course_id.nunique()} courses, {df.University.nunique()} universities")
print(df.head().to_string())


# ============================================================================
# 1. Descriptives + normality
# ============================================================================

for name, x in [('Student mean', hm), ('Student median', hmed), ('GPT-4o', g)]:
    sw = stats.shapiro(x)
    print(f"{name:15s} mean = {x.mean():.2f}, SD = {x.std(ddof=1):.2f}, "
          f"Shapiro–Wilk W = {sw.statistic:.3f}, p = {sw.pvalue:.4f}")
print(f"\nStudent median >= 90 on {(hmed >= 90).sum()} of {n} exams (ceiling check)")


# ============================================================================
# 2. Paired comparisons: t-test, Wilcoxon, effect sizes, CIs (→ Table 4)
# ============================================================================

def rank_biserial(d):
    """Matched-pairs rank-biserial correlation from paired differences d."""
    nz = d[d != 0]
    r = stats.rankdata(np.abs(nz))
    Wp, Wm = r[nz > 0].sum(), r[nz < 0].sum()
    return (Wp - Wm) / (Wp + Wm), Wp, Wm, len(nz)

def hodges_lehmann(d):
    i, j = np.triu_indices(len(d))
    return np.median((d[i] + d[j]) / 2)

def boot_ci(d, fn, B=B):
    vals = np.array([fn(rng.choice(d, len(d))) for _ in range(B)])
    return np.percentile(vals, [2.5, 97.5])

def paired_report(label, student, gpt):
    d = student - gpt                      # positive = students higher
    print(f"=== {label} (student − GPT) ===")
    print(f"  mean diff = {d.mean():.2f}  (SD of diffs = {d.std(ddof=1):.2f})")

    # paired t + CI for mean difference
    t = stats.ttest_rel(student, gpt)
    ci = t.confidence_interval(0.95)
    print(f"  paired t({n-1}) = {t.statistic:.2f}, p = {t.pvalue:.3f}, 95% CI [{ci.low:.2f}, {ci.high:.2f}]")

    # Cohen's dz
    dz = d.mean() / d.std(ddof=1)
    lo, hi = boot_ci(d, lambda s: s.mean() / s.std(ddof=1))
    print(f"  Cohen's dz = {dz:.2f}, 95% CI [{lo:.2f}, {hi:.2f}]")

    # Wilcoxon signed-rank (zeros dropped, normal approximation)
    w = stats.wilcoxon(student, gpt)
    rb, Wp, Wm, nz = rank_biserial(d)
    mu = nz * (nz + 1) / 4
    sig = np.sqrt(nz * (nz + 1) * (2 * nz + 1) / 24)
    z = (Wp - mu) / sig
    print(f"  Wilcoxon: n_nonzero = {nz} (zeros dropped = {n - nz}), "
          f"W+ = {Wp:.1f}, W- = {Wm:.1f}, z = {z:.2f}, p = {w.pvalue:.3f}")
    lo, hi = boot_ci(d, lambda s: rank_biserial(s)[0])
    print(f"  rank-biserial r = {rb:.2f}, 95% CI [{lo:.2f}, {hi:.2f}]")

    # Hodges–Lehmann median difference
    hl = hodges_lehmann(d)
    lo, hi = boot_ci(d, hodges_lehmann, B=2000)
    print(f"  Hodges–Lehmann median diff = {hl:.1f}, 95% CI [{lo:.1f}, {hi:.1f}]")

    print(f"  exams: students higher = {(d > 0).sum()}, GPT higher = {(d < 0).sum()}, tie = {(d == 0).sum()}\n")

paired_report('Student mean vs GPT-4o', hm, g)
paired_report('Student median vs GPT-4o', hmed, g)


# ============================================================================
# 3. Correlations with CIs
# ============================================================================

def corr_report(label, x, y):
    r, p = stats.pearsonr(x, y)
    se = 1 / np.sqrt(n - 3)
    lo, hi = np.tanh(np.arctanh(r) + np.array([-1.96, 1.96]) * se)   # Fisher z
    rho, prho = stats.spearmanr(x, y)
    bs = [stats.spearmanr(x[i], y[i])[0] for i in (rng.choice(n, n) for _ in range(2000))]
    print(f"=== {label} ===")
    print(f"  Pearson r = {r:.3f}, 95% CI [{lo:.3f}, {hi:.3f}], p = {p:.3f}, r² = {r**2:.3f}")
    print(f"  Spearman ρ = {rho:.3f}, 95% CI [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}], p = {prho:.3f}\n")

corr_report('GPT-4o vs student mean', hm, g)
corr_report('GPT-4o vs student median', hmed, g)


# ============================================================================
# 4. Variance by exam difficulty — Brown–Forsythe (Section 3.1)
# ============================================================================

lo_g, hi_g = g[hm < 70], g[hm >= 70]
bf = stats.levene(lo_g, hi_g, center='median')   # center='median' == Brown–Forsythe
print(f"lower-scoring exams  (student mean < 70):  n = {len(lo_g)}, var = {lo_g.var(ddof=1):.1f}, SD = {lo_g.std(ddof=1):.1f}")
print(f"higher-scoring exams (student mean >= 70): n = {len(hi_g)}, var = {hi_g.var(ddof=1):.1f}, SD = {hi_g.std(ddof=1):.1f}")
print(f"variance ratio = {lo_g.var(ddof=1) / hi_g.var(ddof=1):.2f}")
print(f"Brown–Forsythe W = {bf.statistic:.2f}, df = (1, {n - 2}), p = {bf.pvalue:.3f}")


# ============================================================================
# 5. Error type × course level — Cramér's V (Section 3.4, Table 3)
# ============================================================================

# rows: undergraduate, graduate/cross-listed; cols: conceptual, grounding, computational, image
tab = np.array([[143, 81, 48, 21],
                [ 43, 10,  7,  1]])
chi2, p_chi, dof, expected = stats.chi2_contingency(tab)
V = np.sqrt(chi2 / (tab.sum() * (min(tab.shape) - 1)))
print(f"chi² = {chi2:.2f}, df = {dof}, p = {p_chi:.4f}, min expected count = {expected.min():.2f}  (-> use Fisher exact)")
print(f"Cramér's V = {V:.3f}   (Cohen thresholds for df=3: 0.06 / 0.17 / 0.29)")

# Fisher exact for r x c: pip install FisherExact  (or use R: fisher.test(tab))
try:
    from FisherExact import fisher_exact
    print(f"Fisher exact (2x4) p = {fisher_exact(tab):.4f}")
except ImportError:
    print("Fisher exact 2x4: run  !pip install FisherExact  then re-run this cell  (paper reports p = 0.0184)")

# proportion of conceptual errors, UG vs Grad, with Wald 95% CI
p_ug, p_gr = tab[0, 0] / tab[0].sum(), tab[1, 0] / tab[1].sum()
se = np.sqrt(p_ug * (1 - p_ug) / tab[0].sum() + p_gr * (1 - p_gr) / tab[1].sum())
diff = p_gr - p_ug
print(f"conceptual share: UG = {p_ug:.3f}, Grad = {p_gr:.3f}, diff = {diff:.3f}, 95% CI [{diff - 1.96*se:.3f}, {diff + 1.96*se:.3f}]")


# ============================================================================
# 6. Clustering sensitivity — exams nested in courses / universities
# ============================================================================

df['d_mean'] = hm - g
df['d_med'] = hmed - g

def icc1(values, groups):
    """One-way random-effects ICC(1) with unbalanced-group correction."""
    grp = [values[groups == k] for k in np.unique(groups)]
    k = len(grp); N = len(values); grand = values.mean()
    n_i = np.array([len(x) for x in grp])
    msb = sum(len(x) * (x.mean() - grand) ** 2 for x in grp) / (k - 1)
    msw = sum(((x - x.mean()) ** 2).sum() for x in grp) / (N - k)
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    return max((msb - msw) / (msb + (n0 - 1) * msw), 0.0), k

for lvl in ['course_id', 'University']:
    icc, k = icc1(df['d_mean'].values, df[lvl].values)
    kw = stats.kruskal(*[v.values for _, v in df.groupby(lvl)['d_mean']])
    print(f"{lvl:10s}: k = {k:2d}, ICC(1) of paired diff = {icc:.3f}, "
          f"Kruskal–Wallis H({k-1}) = {kw.statistic:.1f}, p = {kw.pvalue:.3f}")

# (a) cluster bootstrap: resample courses with replacement
print("\nCluster bootstrap (resampling courses):")
for lab, col in [('student mean − GPT', 'd_mean'), ('student median − GPT', 'd_med')]:
    cl = [v.values for _, v in df.groupby('course_id')[col]]
    bs = np.array([np.concatenate([cl[i] for i in rng.choice(len(cl), len(cl))]).mean() for _ in range(B)])
    p_bs = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    print(f"  {lab:22s}: 95% CI [{np.percentile(bs, 2.5):.1f}, {np.percentile(bs, 97.5):.1f}], two-sided p ≈ {p_bs:.3f}")

# (b) course-level aggregation
print("\nCourse-level averages (k = courses):")
agg = df.groupby('course_id')[['hm', 'hmed', 'gpt']].mean()
for lab, col in [('student mean − GPT', 'hm'), ('student median − GPT', 'hmed')]:
    w = stats.wilcoxon(agg[col], agg['gpt']); t = stats.ttest_rel(agg[col], agg['gpt'])
    print(f"  {lab:22s}: k = {len(agg)}, mean diff = {(agg[col] - agg['gpt']).mean():.1f}, "
          f"Wilcoxon p = {w.pvalue:.3f}, paired t p = {t.pvalue:.3f}")


# ============================================================================
# 7. Figures (3–7)
# ============================================================================

# Figure style matches the submitted manuscript. Shapiro–Wilk p-values and the A–F labels are computed from the data, not hard-coded. Files are written to `SAVE_DIR` (set in the Configuration cell).

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

def save(fig, name):
    fig.savefig(SAVE_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(SAVE_DIR / f"{name}.pdf", bbox_inches="tight")


# ============================================================================
# Figure 3 — GPT-4o vs student scores, with exams A–F highlighted
# ============================================================================

# A–C: the three exams with the largest student-minus-GPT gap (for Thermodynamics II, the exam with GPT-4o's lowest score); D–F: the three exams with the largest GPT-minus-student gap. Selection is computed here so it stays in sync with Table 2.

df['diff_mean'] = df['hm'] - df['gpt']

# --- select A–C (students higher) and D–F (GPT higher), one exam per course ---
def top_per_course(frame, ascending):
    out = []
    for _, row in frame.sort_values('diff_mean', ascending=ascending).iterrows():
        if row['Course'] not in [r['Course'] for r in out]:
            out.append(row)
        if len(out) == 3:
            break
    return out

students_higher = top_per_course(df[df['diff_mean'] > 0], ascending=False)
# Thermodynamics II: Table 2 uses the exam with GPT-4o's lowest score rather than the largest gap
for i, r in enumerate(students_higher):
    if r['Course'] == 'Thermodynamics II':
        students_higher[i] = df[df['Course'] == 'Thermodynamics II'].sort_values('gpt').iloc[0]
gpt_higher = top_per_course(df[df['diff_mean'] < 0], ascending=True)

# label order follows Table 2: within each group, descending student mean
students_higher = sorted(students_higher, key=lambda r: -r['hm'])
gpt_higher = sorted(gpt_higher, key=lambda r: -r['hm'])
highlights = dict(zip('ABC', students_higher)) | dict(zip('DEF', gpt_higher))
print("Selected exams (should match Table 2):")
for k, r in highlights.items():
    print(f"  {k}: {r['Course']:35s} mean {r['hm']:5.1f}  median {r['hmed']:5.1f}  GPT {r['gpt']:5.1f}")

# --- plot ---
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16,
                     'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 13})

# per-(label, panel) annotation offsets; default (8, -1, left, top).
# In the MEAN panel (idx 0) D(68, 96) and E(67.6, 95) nearly coincide, so their labels are pushed apart.
default_off = (8, -1, 'left', 'top')
label_off = {('D', 0): (9, 2, 'left', 'bottom'),
             ('E', 0): (-9, -7, 'right', 'top')}

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
panels = [('Human (mean)', 'hm', 0, '(a)'), ('Human (median)', 'hmed', 1, '(b)')]
for xlabel, col, idx, corner in panels:
    ax = axes[idx]
    ax.scatter(df[col], df['gpt'], alpha=0.6, s=40, color='dimgray', edgecolor='black', linewidth=0.3)
    for lab, r in highlights.items():
        c = 'red' if lab in 'ABC' else 'blue'
        ax.scatter(r[col], r['gpt'], s=120, color=c, marker='o', edgecolor='black', linewidth=1.5, alpha=0.9, zorder=5)
        dx, dy, ha, va = label_off.get((lab, idx), default_off)
        ax.annotate(lab, xy=(r[col], r['gpt']), xytext=(dx, dy), textcoords='offset points',
                    ha=ha, va=va, color=c, fontsize=13, fontweight='bold', zorder=6)
    ax.plot([0, 100], [0, 100], '--', color='black', alpha=0.4)
    ax.set_xlabel(xlabel); ax.set_ylabel('GPT-4o')
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.text(0.02, 0.97, corner, transform=ax.transAxes, fontsize=16, va='top', ha='left')
    ax.grid(True, alpha=0.3)

# legend only on panel (a), lower-left (empty region)
red_patch = axes[0].scatter([], [], c='red', s=100, label='GPT poor, students better')
blue_patch = axes[0].scatter([], [], c='blue', s=100, label='GPT good, students poor')
axes[0].legend(handles=[red_patch, blue_patch], loc='lower left', framealpha=0.9)
plt.tight_layout(); save(fig, 'Figure3_scatter'); plt.close(fig)


# ============================================================================
# Figure 4 — Score distributions with Shapiro–Wilk p-values
# ============================================================================

def fmt_p(p):
    return 'p < 0.001' if p < 0.001 else f'p = {p:.2f}'

cols = [('hm', '(a)'), ('hmed', '(b)'), ('gpt', '(c)')]
bin_edges = np.arange(20, 101, 10)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
max_y = 0
for ax, (col, lab) in zip(axes, cols):
    data = df[col]
    p = stats.shapiro(data).pvalue          # computed, not hard-coded
    sns.histplot(data, kde=True, bins=bin_edges, ax=ax)
    max_y = max(max_y, ax.get_ylim()[1])
    ax.set_xlim(20, 100)
    ax.set_xlabel('Score', fontsize=16)
    ax.set_ylabel('Frequency', fontsize=16)
    ax.tick_params(axis='both', labelsize=18)
    ax.text(0.02, 0.95, lab, transform=ax.transAxes, fontsize=18, va='top', ha='left')
    ax.text(0.18, 0.60, f'Shapiro {fmt_p(p)}', transform=ax.transAxes, fontsize=18, va='top', ha='left', color='black')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    print(f'{col:5s} Shapiro {fmt_p(p)}')
for ax in axes:
    ax.set_ylim(0, max_y)
plt.tight_layout(); save(fig, 'Figure4_shapiro'); plt.close(fig)


# ============================================================================
# Figure 5 — Paired differences (student − GPT-4o)
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (d, xlabel, lab) in zip(axes, [(df['hm'] - df['gpt'], 'Student Mean – GPT-4o', '(a)'),
                                       (df['hmed'] - df['gpt'], 'Student Median – GPT-4o', '(b)')]):
    sns.histplot(d, kde=True, stat='percent', ax=ax)
    ax.set_xlabel(xlabel); ax.set_ylabel('Frequency (%)')
    ax.text(0.02, 0.98, lab, transform=ax.transAxes, fontsize=12, va='top', ha='left')
    print(f'{xlabel}: mean = {d.mean():.2f}, median = {d.median():.2f}')   # for the caption
plt.tight_layout(); save(fig, 'Figure5_differences'); plt.close(fig)


# ============================================================================
# Figures 6 and 7 — Error-type distributions (from Table 3)
# ============================================================================

cats = ['Conceptual', 'Grounding', 'Computational', 'Image\nunderstanding']
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
tab = np.array([[143, 81, 48, 21], [43, 10, 7, 1]])   # UG, Grad

def pie(ax, counts, title):
    ax.pie(counts, labels=cats, colors=colors, startangle=90, counterclock=False,
           autopct=lambda pct: f'{pct:.1f}%\n({int(round(pct * counts.sum() / 100))})',
           wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}, textprops={'fontsize': 12})
    ax.set_title(title, fontsize=14)

# Figure 6: all errors
fig, ax = plt.subplots(figsize=(6.5, 6.5))
pie(ax, tab.sum(axis=0), f'All exams (n = {tab.sum()} errors)')
plt.tight_layout(); save(fig, 'Fig6_errors_all'); plt.close(fig)

# Figure 7: by course level
fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
pie(axes[0], tab[0], f'Undergraduate courses (n = {tab[0].sum()})')
pie(axes[1], tab[1], f'Graduate / cross-listed courses (n = {tab[1].sum()})')
plt.tight_layout(); save(fig, 'Fig7_errors_by_level'); plt.close(fig)
