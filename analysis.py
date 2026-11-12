"""Occupational injuries among foreign workers in Japan, 2023-2025:
industry- and age-standardized incidence ratios (SIRs) by residence status.

All inputs are public aggregate statistics stored in ./data (see README.md).
Run:  python analysis.py     -> tables in ./output/*.csv, figures in ./output/*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

try:  # optional cross-check of the exact Poisson interval with a standard package
    from statsmodels.stats.rates import confint_poisson
except Exception:  # pragma: no cover
    confint_poisson = None

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE / "data", HERE / "output"
OUT.mkdir(exist_ok=True)
AGE = ["15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65+"]
K5 = ["mfg", "con", "com", "hlt", "hos"]
K6 = K5 + ["oth"]
YEARS = [2023, 2024, 2025]
STATUS = ["ssw", "titp", "status_based", "designated_activities", "other_professional", "engineer_humanities", "outside_status"]
LABEL = {"ssw": "Specified skilled worker", "titp": "Technical intern training", "status_based": "Status-based residents",
         "designated_activities": "Designated activities", "other_professional": "Other professional/technical",
         "engineer_humanities": "Engineer/Humanities/Intl. services", "outside_status": "Permitted outside-status work",
         "titp_ssw": "TITP + SSW combined"}
WORKER_STATUS = {"titp", "ssw", "engineer_humanities", "other_professional"}  # residents are workers by definition


def sir_ci(obs, exp):
    """SIR with exact (Garwood) 95% CI for a Poisson count."""
    lo = chi2.ppf(0.025, 2 * obs) / 2 / exp if obs > 0 else 0.0
    hi = chi2.ppf(0.975, 2 * obs + 2) / 2 / exp
    return obs / exp, lo, hi


def vec(df, col):
    return df.set_index("age")[col].reindex(AGE).to_numpy(float)


# ---------------------------------------------------------------- load
inj = pd.read_csv(DATA / "ref_injuries.csv")
emp = pd.read_csv(DATA / "ref_employees.csv")
off = pd.read_csv(DATA / "official_rates.csv").set_index(["year", "industry"])["rate_per_1000"]
fw8 = pd.read_csv(DATA / "foreign_all_by_industry8.csv")
ref8 = pd.read_csv(DATA / "ref_injuries_industry8.csv").set_index(["year", "industry8"])["all_worker_injuries"]
fws = pd.read_csv(DATA / "foreign_workers_by_status_industry.csv").set_index(["year", "status", "industry"])["workers"]
fis = pd.read_csv(DATA / "foreign_injuries_by_status_industry.csv")
res = pd.read_csv(DATA / "residents_by_status_sex_age.csv")
epr = pd.read_csv(DATA / "employee_population_ratio_2025.csv")
isa = pd.read_csv(DATA / "isa_midyear_residents.csv").set_index(["year", "status"])["residents_end_of_june"]
acc = pd.read_csv(DATA / "accident_type_2025.csv")
W = {s: vec(epr[epr.sex == s], "employee_population_ratio") for s in "MF"}

# ---------------------------------------------------------------- reference rates (all workers)
rate, rate0, rate_sex = {}, {}, {}
for y in YEARS:
    iy, ey = inj[inj.year == y], emp[emp.year == y]
    inj_k = {k: vec(iy[iy.industry == k].groupby("age", as_index=False).injuries.sum(), "injuries") for k in K6}
    emp_k = {k: vec(ey[(ey.industry == k) & (ey.sex == "T")], "employees") for k in K6}
    tot_inj = ref8[(y, "tot")]
    # level of each stratum = officially published rate (denominator: employees excl. executives);
    # the residual stratum rate is derived from the published total rate
    n_off = {k: inj_k[k].sum() / off[(y, k)] * 1000 for k in K5}
    r0 = {k: off[(y, k)] / 1000 for k in K5}
    r0["oth"] = inj_k["oth"].sum() / (tot_inj / off[(y, "tot")] * 1000 - sum(n_off.values()))
    cal = {k: r0[k] / (inj_k[k].sum() / emp_k[k].sum()) for k in K6}
    rate0[y] = r0
    rate[y] = {k: inj_k[k] / emp_k[k] * cal[k] for k in K6}  # age-specific, calibrated
    if set(iy.sex) >= {"M", "F"}:
        rs = {}
        for k in K6:
            m = vec(iy[(iy.industry == k) & (iy.sex == "M")], "injuries") / vec(ey[(ey.industry == k) & (ey.sex == "M")], "employees")
            fden = vec(ey[(ey.industry == k) & (ey.sex == "F")], "employees")
            f = np.divide(vec(iy[(iy.industry == k) & (iy.sex == "F")], "injuries"), fden, out=np.zeros(11), where=fden > 0)
            rs[k] = np.array([m, f]) * cal[k]
        rate_sex[y] = rs


# ---------------------------------------------------------------- age/sex structure of foreign workers
def structure(y, status, weight=True):
    """Joint sex x age distribution (2 x 11, sums to 1) of workers with a given status."""
    def one(g):
        d = res[(res.year == y) & (res.status == g)]
        x = np.array([vec(d[d.sex == s], "residents") for s in "MF"])
        if weight and g not in WORKER_STATUS:  # residents -> workers, using employee/population ratios
            x = x * np.array([W["M"], W["F"]])
        return x / x.sum()
    if status == "outside_status":  # students + dependents
        n, ns = fws[(y, status, "total")], fws[(y, "student_workers", "total")]
        return (ns * one("student") + (n - ns) * one("dependent")) / n
    return one(status)


# ---------------------------------------------------------------- all foreign workers: 8 industries, published rates
def expected_all(y):
    d = fw8[fw8.year == y].set_index("industry8")
    g = lambda k: off[(y, k)]
    n = {k: ref8[(y, k)] / g(k) * 1000 for k in ["mfg", "con", "trn", "com", "hlt", "hos"]}
    n_agf = ref8[(y, "agf")] / ((0.9 * g("agr") + 0.1 * g("fsh")) / 1000)  # approximation for agriculture/forestry/fishery employees
    inj_named = sum(ref8[(y, k)] for k in ["mfg", "con", "trn", "agf", "com", "hlt", "hos"])
    rest = (ref8[(y, "tot")] - inj_named) / (ref8[(y, "tot")] / g("tot") * 1000 - sum(n.values()) - n_agf)
    w = d.foreign_workers
    e = {"mfg": w["mfg"] * g("mfg"), "con": w["con"] * g("con"), "trn": w["trn"] * g("trn"),
         "agf": w["_agr_workers"] * g("agr") + w["_fsh_workers"] * g("fsh"), "com": w["com"] * g("com"),
         "hlt": w["hlt"] * g("hlt"), "hos": w["hos"] * g("hos"), "oth": w["oth"] * rest * 1000}
    e = {k: v / 1000 for k, v in e.items()}
    disp = w["_dispatch_agency_workers"]  # upper-bound sensitivity: all agency workers placed in manufacturing
    e_disp = sum(e.values()) + disp * (g("mfg") / 1000 - rest)
    obs = d.loc[list(e), "foreign_injuries"]
    return obs.sum(), sum(e.values()), e_disp, {k: (obs[k], e[k]) for k in e}, w.loc[list(e)].sum()


# ---------------------------------------------------------------- by status
def workers(y, s):
    n = [fws[(y, s, k)] for k in K5]
    return n + [fws[(y, s, "total")] - sum(n)]


def expected_status(y, s, weight=True):
    n, p = workers(y, s), structure(y, s, weight)
    pa = p.sum(0)
    out = {"n": sum(n), "e_ind": sum(n[i] * rate0[y][k] for i, k in enumerate(K6)),
           "e_age": sum(n[i] * (pa * rate[y][k]).sum() for i, k in enumerate(K6)),
           "e_age_by_ind": {k: n[i] * (pa * rate[y][k]).sum() for i, k in enumerate(K6)}, "n_by_ind": dict(zip(K6, n))}
    if y in rate_sex:
        rs = rate_sex[y]
        out["e_sex"] = sum(n[i] * (p * rs[k]).sum() for i, k in enumerate(K6))  # sex mix independent of industry
        pm, pf = p[0] / p[0].sum(), p[1] / p[1].sum()                      # construction all male, others rebalanced
        m_rest, f_rest = max(p[0].sum() * sum(n) - n[1], 0), p[1].sum() * sum(n)
        q = np.array([pm * m_rest, pf * f_rest]) / (m_rest + f_rest)
        out["e_sex_con_male"] = n[1] * (pm * rs["con"][0]).sum() + sum(n[i] * (q * rs[k]).sum() for i, k in enumerate(K6) if i != 1)
    return out


obs_status = fis.groupby(["year", "status"]).injuries.sum()
obs_si = fis.set_index(["year", "status", "industry9"]).injuries
rows, pooled, yearly_all, ind_rows = [], {}, [], []
age_factor = {}
for y in YEARS:
    tot_ei = tot_ea = tot_ea_raw = 0.0
    rest_titp = fws[(y, "titp", "total")] - sum(fws[(y, "titp", k)] for k in K5)
    rest_ssw = fws[(y, "ssw", "total")] - sum(fws[(y, "ssw", k)] for k in K5)
    d8 = fw8[fw8.year == y].set_index("industry8").foreign_workers
    agri_n = d8["_agr_workers"] + d8["_fsh_workers"]
    for s in STATUS:
        e, er, o = expected_status(y, s), expected_status(y, s, weight=False), int(obs_status[(y, s)])
        pa = structure(y, s).sum(0)
        ag = agri_n * {"titp": rest_titp, "ssw": rest_ssw}.get(s, 0) / (rest_titp + rest_ssw)
        e_agri = e["e_age"] + ag * (off[(y, "agr")] / 1000 - rate0[y]["oth"]) * ((pa * rate[y]["oth"]).sum() / rate0[y]["oth"])
        scale_isa = isa[(y, s)] / e["n"] if (y, s) in isa.index else 1.0  # mid-year immigration stock as denominator
        rec = dict(year=y, status=s, obs=o, workers=e["n"], crude_ratio=o / e["n"] * 1000 / off[(y, "tot")], e_ind=e["e_ind"], e_age=e["e_age"],
                   e_age_raw=er["e_age"], e_agri=e_agri, e_isa=e["e_age"] * scale_isa, isa_workers=e["n"] * scale_isa,
                   e_sex=e.get("e_sex", np.nan), e_sex_con_male=e.get("e_sex_con_male", np.nan),
                   under35=pa[:4].sum() * 100, male=structure(y, s)[0].sum() * 100)
        rows.append(rec)
        tot_ei += e["e_ind"]; tot_ea += e["e_age"]; tot_ea_raw += er["e_age"]
        for k9, k in [("mfg", "mfg"), ("con", "con"), ("com", "com"), ("hlt", "hlt"), ("hos", "hos")]:
            ind_rows.append(dict(year=y, status=s, industry=k, obs=int(obs_si[(y, s, k9)]), workers=e["n_by_ind"][k], e_age=e["e_age_by_ind"][k],
                                 e_ind=e["n_by_ind"][k] * rate0[y][k]))
    f, f_raw = tot_ea / tot_ei, tot_ea_raw / tot_ei
    age_factor[y] = f
    O, E8, E8d, by8, n_all = expected_all(y)
    yearly_all.append(dict(year=y, obs=O, workers=n_all, crude_ratio=O / n_all * 1000 / off[(y, "tot")], e_ind=E8, e_age=E8 * f, e_age_raw=E8 * f_raw,
                           e_ind_disp=E8d, e_age_disp=E8d * f, age_factor=f, **{f"obs_{k}": v[0] for k, v in by8.items()}, **{f"exp_{k}": v[1] for k, v in by8.items()}))
T = pd.DataFrame(rows)
A = pd.DataFrame(yearly_all)
I = pd.DataFrame(ind_rows)


def summarise(d, cols):
    o = d.obs.sum()
    out = {"obs": int(o)}
    for c in cols:
        dd = d[d[c].notna()]
        if len(dd):
            out["sir_" + c[2:]] = [round(v, 3) for v in sir_ci(dd.obs.sum(), dd[c].sum())]
    return out


ECOLS = ["e_ind", "e_age", "e_age_raw", "e_agri", "e_isa", "e_sex", "e_sex_con_male"]
R = {"all": {"pooled": summarise(A, ["e_ind", "e_age", "e_age_raw", "e_ind_disp", "e_age_disp"]),
             "by_year": {int(r.year): summarise(A[A.year == r.year], ["e_ind", "e_age", "e_ind_disp", "e_age_disp"]) for r in A.itertuples()},
             "crude_ratio": {int(r.year): round(r.crude_ratio, 3) for r in A.itertuples()},
             "crude_ratio_pooled": round(A.obs.sum() / A.workers.sum() / (ref8.xs("tot", level=1).sum() / sum(ref8[(y, "tot")] / off[(y, "tot")] for y in YEARS)) * 1000, 3)},
     "status": {}, "status_industry": {}, "accident_type_share_2025": {}}
ref_rate3 = ref8.xs("tot", level=1).sum() / sum(ref8[(y, "tot")] / off[(y, "tot")] * 1000 for y in YEARS) * 1000
for s in STATUS + ["titp_ssw"]:
    d = T[T.status.isin(["titp", "ssw"])] if s == "titp_ssw" else T[T.status == s]
    r = summarise(d, ECOLS)
    r["crude_ratio"] = round(d.obs.sum() / d.workers.sum() * 1000 / ref_rate3, 3)
    r["crude_ratio_isa"] = round(d.obs.sum() / d.isa_workers.sum() * 1000 / ref_rate3, 3)
    d2 = d[d.year.isin(rate_sex)]
    r["sex_period"] = summarise(d2, ["e_age", "e_sex", "e_sex_con_male"])
    r["by_year"] = {int(y): {"crude": round(g.obs.sum() / g.workers.sum() * 1000 / off[(y, "tot")], 2), **summarise(g, ["e_ind", "e_age", "e_isa"])} for y, g in d.groupby("year")}
    R["status"][s] = r
    di = I[I.status.isin(["titp", "ssw"])] if s == "titp_ssw" else I[I.status == s]
    R["status_industry"][s] = {k: {"obs": int(g.obs.sum()), "rate_per_1000": round(g.obs.sum() / g.workers.sum() * 1000, 2),
                                   "sir_ind": [round(v, 2) for v in sir_ci(g.obs.sum(), g.e_ind.sum())],
                                   "sir_age": [round(v, 2) for v in sir_ci(g.obs.sum(), g.e_age.sum())]} for k, g in di.groupby("industry")}
for c in [c for c in acc.columns if c != "accident_type"]:
    tot = {"all_workers": ref8[(2025, "tot")]}.get(c, obs_status.get((2025, c), np.nan))
    R["accident_type_share_2025"][c] = {t: round(v / tot * 100, 1) for t, v in zip(acc.accident_type, acc[c])}
R["structure"] = {int(y): {s: {"under35_pct": round(float(g.under35.iloc[0]), 1), "male_pct": round(float(g.male.iloc[0]), 1)} for s, g in T[T.year == y].groupby("status")} for y in YEARS}
R["age_factor"] = {int(k): round(v, 3) for k, v in age_factor.items()}
R["reference_rate_2025"] = {k: [round(v * 1000, 2) for v in rate[2025][k]] for k in K6}
if confint_poisson is not None:  # cross-check one interval with statsmodels
    o, e = R["all"]["pooled"]["obs"], A.e_age.sum()
    lo, hi = confint_poisson(o, e, method="exact-c")
    R["check_statsmodels_all_pooled_age"] = [round(o / e, 3), round(lo, 3), round(hi, 3)]
(OUT / "results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float))
T.to_csv(OUT / "table_status_year.csv", index=False)
A.to_csv(OUT / "table_all_year.csv", index=False)
I.to_csv(OUT / "table_status_industry_year.csv", index=False)


# ---------------------------------------------------------------- medical vs welfare within "medical, health care and welfare"
hw = pd.read_csv(DATA / "ref_health_welfare_sub.csv")
fhw = pd.read_csv(DATA / "foreign_workers_health_welfare.csv").set_index("year")
hw_rows = []
for y in YEARS:
    d = hw[hw.year == y]
    iy, ey = inj[(inj.year == y) & (inj.industry == "hlt")], emp[(emp.year == y) & (emp.industry == "hlt") & (emp.sex == "T")]
    cal_h = off[(y, "hlt")] / 1000 / (iy.injuries.sum() / ey.employees.sum())  # same level adjustment as the parent stratum
    def sub_rate(sub, sex="T"):
        num = d[(d["sub"] == sub) & (d.measure == "injuries")]
        num = num[num.sex == sex] if sex in set(num.sex) else num.groupby("age", as_index=False).value.sum()
        den = d[(d["sub"] == sub) & (d.measure == "employees") & (d.sex == sex)]
        den_v = vec(den, "value")
        return np.divide(vec(num, "value"), den_v, out=np.zeros(11), where=den_v > 0) * cal_h
    r_med, r_wel = sub_rate("medical"), sub_rate("welfare")
    share_w = fhw.loc[y, "welfare"] / (fhw.loc[y, "welfare"] + fhw.loc[y, "medical"])
    for s in STATUS:
        n, p = fws[(y, s, "hlt")], structure(y, s)
        pa = p.sum(0)
        rec = dict(year=y, status=s, obs=int(obs_si[(y, s, "hlt")]), workers=n, e_main=n * (pa * rate[y]["hlt"]).sum(),
                   e_mix=n * (share_w * (pa * r_wel).sum() + (1 - share_w) * (pa * r_med).sum()), e_welfare=n * (pa * r_wel).sum(),
                   e_welfare_female=np.nan, rate_welfare=(d[(d["sub"] == "welfare") & (d.measure == "injuries")].value.sum() / d[(d["sub"] == "welfare") & (d.measure == "employees") & (d.sex == "T")].value.sum()) * cal_h * 1000,
                   rate_medical=(d[(d["sub"] == "medical") & (d.measure == "injuries")].value.sum() / d[(d["sub"] == "medical") & (d.measure == "employees") & (d.sex == "T")].value.sum()) * cal_h * 1000, welfare_share=share_w)
        if "F" in set(d.sex):
            rec["e_welfare_female"] = n * ((p[1] / p[1].sum()) * sub_rate("welfare", "F")).sum()
        hw_rows.append(rec)
HW = pd.DataFrame(hw_rows)
HW.to_csv(OUT / "table_health_welfare.csv", index=False)
R["health_welfare"] = {"reference_rate_per_1000": {int(y): {"welfare": round(float(g.rate_welfare.iloc[0]), 2), "medical": round(float(g.rate_medical.iloc[0]), 2), "welfare_share_foreign": round(float(g.welfare_share.iloc[0]), 3)} for y, g in HW.groupby("year")}}
for s in STATUS + ["titp_ssw"]:
    g = HW[HW.status.isin(["titp", "ssw"])] if s == "titp_ssw" else HW[HW.status == s]
    R["health_welfare"][s] = {"obs": int(g.obs.sum()), "rate_per_1000": round(g.obs.sum() / g.workers.sum() * 1000, 2), **{c.replace("e_", "sir_", 1): [round(v, 2) for v in sir_ci(g.obs.sum(), g[c].sum())] for c in ["e_main", "e_mix", "e_welfare"]}}
    g2 = g[g.e_welfare_female.notna()]
    R["health_welfare"][s]["sir_welfare_female_2023_24"] = [round(v, 2) for v in sir_ci(g2.obs.sum(), g2.e_welfare_female.sum())]
    R["health_welfare"][s]["sir_welfare_2023_24"] = [round(v, 2) for v in sir_ci(g2.obs.sum(), g2.e_welfare.sum())]
(OUT / "results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float))


# ---------------------------------------------------------------- additional sensitivity analyses (revision 1)
SENS = {}
# (a) age composition differing by industry within a status: move delta of the construction workforce between ages 20-24 and 30-34,
#     with the opposite shift in manufacturing so that the status-wide age distribution is unchanged
i20, i30 = AGE.index("20-24"), AGE.index("30-34")
COMB = {}
for s in ["titp", "ssw"]:
    SENS.setdefault("age_by_industry", {})[s] = {}
    for delta in [0.0, 0.08, -0.08, 0.12, -0.12]:
        o_c = o_m = e_c = e_m = 0.0
        for y in YEARS:
            n, pa = workers(y, s), structure(y, s).sum(0)
            pc, pm = pa.copy(), pa.copy()
            pc[i20] += delta; pc[i30] -= delta
            dm = delta * n[1] / n[0]
            pm[i20] -= dm; pm[i30] += dm
            e_c += n[1] * (pc * rate[y]["con"]).sum(); e_m += n[0] * (pm * rate[y]["mfg"]).sum()
            o_c += obs_si[(y, s, "con")]; o_m += obs_si[(y, s, "mfg")]
        SENS["age_by_industry"][s]["%+.2f" % delta] = {"con": round(o_c / e_c, 2), "mfg": round(o_m / e_m, 2)}
        for k, o, e in [("con", o_c, e_c), ("mfg", o_m, e_m)]:
            COMB.setdefault("%+.2f" % delta, {}).setdefault(k, [0.0, 0.0]); COMB["%+.2f" % delta][k][0] += o; COMB["%+.2f" % delta][k][1] += e
SENS["age_by_industry"]["titp_ssw"] = {d: {k: round(v[0] / v[1], 2) for k, v in x.items()} for d, x in COMB.items()}
# (b) denominators of all foreign workers: mean of the end-of-October stocks of the current and previous year, and +/-10%
N_OCT = {2022: 1822725, 2023: 2048675, 2024: 2302587, 2025: 2571037}
e_mid = sum(r.e_age * (N_OCT[r.year - 1] + N_OCT[r.year]) / 2 / N_OCT[r.year] for r in A.itertuples())
SENS["all_denominator"] = {"main": round(A.obs.sum() / A.e_age.sum(), 3), "mean_of_adjacent_october_stocks": [round(v, 3) for v in sir_ci(A.obs.sum(), e_mid)],
                           "denominator_plus10pct": round(A.obs.sum() / (A.e_age.sum() * 1.1), 3), "denominator_minus10pct": round(A.obs.sum() / (A.e_age.sum() * 0.9), 3)}
# (c) all foreign workers: sum of status-specific expected numbers (six industries) instead of eight industries x pooled age factor
#     observed numbers are restricted to injuries with known status (55 injuries with unknown status are excluded)
SENS["all_six_strata"] = {"obs_known_status": int(T.obs.sum()), "expected": round(T.e_age.sum(), 1),
                          "sir_ind": [round(v, 3) for v in sir_ci(T.obs.sum(), T.e_ind.sum())], "sir_age": [round(v, 3) for v in sir_ci(T.obs.sum(), T.e_age.sum())],
                          "by_year_age": {int(y): round(g.obs.sum() / g.e_age.sum(), 3) for y, g in T.groupby("year")}}
# (b2) denominator timing: weight w on the previous end-of-October number (0 = main analysis, 1/2 = simple mean, 1/3 = linear interpolation to end of June)
SENS["all_denominator"]["by_weight_on_previous_october"] = {"%.2f" % w: round(A.obs.sum() / sum(r.e_age * (w * N_OCT[r.year - 1] + (1 - w) * N_OCT[r.year]) / N_OCT[r.year] for r in A.itertuples()), 3) for w in [0, 0.25, 1 / 3, 0.5]}
# (d) overall estimate under alternative age factors applied to the eight-industry expected number
SENS["all_age_factor"] = {"%.2f" % f: round(A.obs.sum() / (A.e_ind.sum() * f), 3) for f in [0.85, 0.88, 0.90, 0.95, 1.00]}
# (e) combined scenarios for all foreign workers (revision 3): the least favourable assumptions taken together
_o = A.obs.sum()
SENS["all_combined"] = {
    "agency_in_manufacturing_and_denominator_plus10pct": [round(v, 3) for v in sir_ci(_o, A.e_age_disp.sum() * 1.1)],
    "denominator_plus10pct_and_age_factor_0.95": [round(v, 3) for v in sir_ci(_o, A.e_ind.sum() * 0.95 * 1.1)],
    "agency_in_manufacturing_denominator_plus10pct_and_age_factor_0.95": [round(v, 3) for v in sir_ci(_o, A.e_ind_disp.sum() * 0.95 * 1.1)],
    "agency_in_manufacturing_denominator_plus10pct_and_no_age_adjustment": [round(v, 3) for v in sir_ci(_o, A.e_ind_disp.sum() * 1.1)],
}
# (f) observed and expected numbers, pooled (for Supplementary Table S3)
_t = pd.read_csv(OUT / "table_status_year.csv").groupby("status")[["obs", "e_ind", "e_age"]].sum()
_t.loc["all"] = [A.obs.sum(), A.e_ind.sum(), A.e_age.sum()]
_t["excess"] = _t.obs - _t.e_age
_t.round(1).to_csv(OUT / "table_observed_expected_pooled.csv")
# (g) industry-specific SIRs including the residual industries, pooled (for Supplementary Table S4)
_i = pd.read_csv(OUT / "table_status_industry_year.csv").groupby(["status", "industry"])[["obs", "e_age"]].sum().reset_index()
_tot = pd.read_csv(OUT / "table_status_year.csv").groupby("status")[["obs", "e_age"]].sum()
_res = (_tot - _i.groupby("status")[["obs", "e_age"]].sum()).reset_index(); _res["industry"] = "oth"  # residual = status total minus the five named industries
_i = pd.concat([_i, _res[_res.status.isin(_i.status.unique())]], ignore_index=True)
_c = _i[_i.status.isin(["titp", "ssw"])].groupby("industry")[["obs", "e_age"]].sum().reset_index(); _c["status"] = "titp_ssw"
_i = pd.concat([_i, _c], ignore_index=True)
_i[["sir", "lo", "hi"]] = [[round(v, 2) for v in sir_ci(o, e)] for o, e in zip(_i.obs, _i.e_age)]
_i["e_age"] = _i.e_age.round(1)
_i.to_csv(OUT / "table_status_industry_pooled.csv", index=False)
# (h) age distributions used for standardization, by residence status and year (for Supplementary Table S5)
_rows = []
for _y in YEARS:
    _e = emp[(emp.year == _y) & (emp.sex == "T")].groupby("age").employees.sum().reindex(AGE).to_numpy(float)
    _rows.append(dict(year=_y, group="all_employees", **dict(zip(AGE, (_e / _e.sum() * 100).round(1)))))
    for _s in STATUS:
        _rows.append(dict(year=_y, group=_s, **dict(zip(AGE, (structure(_y, _s).sum(0) * 100).round(1)))))
pd.DataFrame(_rows).to_csv(OUT / "table_age_structure.csv", index=False)
R["sensitivity_revision1"] = SENS
(OUT / "results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float))

# ---------------------------------------------------------------- figures (English)
plt.rcParams["font.size"] = 14
BL, OR, AQ, YL, INK, MUT, SUR = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#1f2328", "#6b7280", "#ffffff"


def style(a):
    a.set_facecolor(SUR)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.spines["left"].set_color("#d0d4d9"); a.spines["bottom"].set_color("#d0d4d9")
    a.tick_params(colors=MUT, labelsize=13); a.grid(axis="y", color="#e8eaed", lw=0.8); a.set_axisbelow(True)


x = np.arange(11)
fig, ax = plt.subplots(1, 2, figsize=(14, 6), facecolor=SUR)
a = ax[0]; style(a)
e25 = emp[(emp.year == 2025) & (emp.sex == "T")].groupby("age").employees.sum().reindex(AGE).to_numpy(float)
# each series differs in colour, marker shape and line style, so that the figures remain readable in greyscale
for nm, p, c, mk, ls in [("All employees in Japan", e25 / e25.sum(), BL, "o", "-"), ("Technical intern training", structure(2025, "titp").sum(0), OR, "s", "--"),
                         ("Specified skilled worker", structure(2025, "ssw").sum(0), AQ, "^", "-."), ("Status-based residents", structure(2025, "status_based").sum(0), YL, "D", ":")]:
    a.plot(x, p * 100, color=c, lw=2.2, marker=mk, ms=7, ls=ls, label=nm)
a.set_xticks(x); a.set_xticklabels(AGE, fontsize=11.5); a.set_ylabel("Share of workers (%)", color=MUT); a.set_xlabel("Age (yr)", color=MUT)
a.legend(frameon=False, fontsize=13.5, handlelength=3.2); a.set_title("A  Age structure of workers, 2025", loc="left", fontsize=16, color=INK)
a = ax[1]; style(a)
for k, nm, c, mk, ls in [("oth", "Other industries", BL, "o", "-"), ("mfg", "Manufacturing", OR, "s", "--"), ("con", "Construction", AQ, "^", "-.")]:
    a.plot(x, rate[2025][k] * 1000, color=c, lw=2.2, marker=mk, ms=7, ls=ls, label=nm)
a.set_xticks(x); a.set_xticklabels(AGE, fontsize=11.5); a.set_ylim(0, None); a.set_ylabel("Injuries per 1,000 workers per year", color=MUT); a.set_xlabel("Age (yr)", color=MUT)
a.legend(frameon=False, fontsize=13.5, loc="upper right", handlelength=3.2); a.set_title("B  Age-specific injury rates of all workers, 2025", loc="left", fontsize=16, color=INK)
plt.tight_layout(); plt.savefig(OUT / "Figure1.png", dpi=300, facecolor=SUR); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(15, 8), facecolor=SUR, gridspec_kw={"width_ratios": [1.25, 1]})
a = ax[0]; style(a)
order = sorted(STATUS, key=lambda s: -R["status"][s]["sir_age"][0])
# rows: all foreign workers, the two statuses combined, then the seven statuses
ALLROW = dict(R["all"]["pooled"], crude_ratio=R["all"]["crude_ratio_pooled"])
rows = [("All foreign workers", ALLROW), ("Technical intern training +\nspecified skilled worker", R["status"]["titp_ssw"])] + [(LABEL[s], R["status"][s]) for s in order]
yy = np.arange(len(rows))[::-1]
OFF = 0.26  # vertical offset so that the three estimates of a status do not overlap
for j, (nm, d) in zip(yy, rows):
    s0, s1, s2 = d["crude_ratio"], d["sir_ind"], d["sir_age"]
    if j % 2 == 0:
        a.axhspan(j - 0.5, j + 0.5, color="#f5f6f8", zorder=0)
    a.scatter([s0], [j + OFF], s=70, facecolor=SUR, edgecolor=INK, linewidth=1.8, marker="D", zorder=3)
    a.errorbar([s1[0]], [j], xerr=[[s1[0] - s1[1]], [s1[2] - s1[0]]], fmt="s", color=BL, mfc=SUR, mew=2, ms=8, lw=2.2, capsize=4, capthick=2, zorder=4)
    a.errorbar([s2[0]], [j - OFF], xerr=[[s2[0] - s2[1]], [s2[2] - s2[0]]], fmt="o", color=OR, ms=8, lw=2.2, capsize=4, capthick=2, zorder=5)
a.scatter([], [], s=70, facecolor=SUR, edgecolor=INK, linewidth=1.8, marker="D", label="Crude rate ratio")
a.errorbar([], [], xerr=[], fmt="s", color=BL, mfc=SUR, mew=2, ms=8, lw=2.2, capsize=4, label="SIR adjusted for industry (95% CI)")
a.errorbar([], [], xerr=[], fmt="o", color=OR, ms=8, lw=2.2, capsize=4, label="SIR adjusted for industry and age (95% CI)")
a.axvline(1, color=MUT, lw=1.2); a.set_yticks(yy); a.set_yticklabels([nm for nm, _ in rows], fontsize=13.5); a.set_xlim(0, 2.2); a.set_ylim(-0.6, len(rows) - 0.4)
a.axhline(len(rows) - 2.5, color=MUT, lw=0.8, ls="--")
a.grid(axis="y", visible=False); a.grid(axis="x", color="#e8eaed", lw=0.8); a.legend(frameon=False, fontsize=13.5, loc="upper center", bbox_to_anchor=(0.35, -0.1), ncol=1, columnspacing=1.2, handletextpad=0.3)
a.set_xlabel("Ratio to all workers (observed / expected)", color=MUT); a.set_title("A  By residence status, 2023-2025 pooled", loc="left", fontsize=16, color=INK)
a = ax[1]; style(a)
series = [("ssw", AQ, LABEL["ssw"], 0.0, "^", "-."), ("status_based", YL, LABEL["status_based"], 0.0, "D", ":"), ("titp", OR, LABEL["titp"], 0.07, "s", "--"), (None, BL, "All foreign workers", -0.07, "o", "-")]
for s, c, nm, dy, mk, ls in series:
    v = [(R["status"][s] if s else R["all"])["by_year"][y]["sir_age"] for y in YEARS]
    a.errorbar(YEARS, [t[0] for t in v], yerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], color=c, lw=2.2, marker=mk, ms=8, ls=ls, capsize=4, capthick=1.8)
    a.plot([2025.06, 2025.2], [v[-1][0], v[-1][0] + dy], color=c, lw=1.2)
    a.text(2025.23, v[-1][0] + dy, nm, va="center", fontsize=14, color=INK)
a.axhline(1, color=MUT, lw=1.2); a.set_xticks(YEARS); a.set_xlim(2022.7, 2028.0); a.set_ylim(0, 2.4)
a.set_ylabel("SIR adjusted for industry and age (95% CI)", color=MUT); a.set_title("B  By year", loc="left", fontsize=16, color=INK)
plt.tight_layout(); plt.savefig(OUT / "Figure2.png", dpi=300, facecolor=SUR); plt.close()

if __name__ == "__main__":
    pr = lambda v: "%.2f (%.2f-%.2f)" % tuple(v)
    print("ALL pooled", {k: (pr(v) if isinstance(v, list) else v) for k, v in R["all"]["pooled"].items()}, "crude", R["all"]["crude_ratio_pooled"])
    for s in STATUS + ["titp_ssw"]:
        d = R["status"][s]
        print(s, "obs", d["obs"], "crude", d["crude_ratio"], "| ind", pr(d["sir_ind"]), "| age", pr(d["sir_age"]), "| agri", pr(d["sir_agri"]),
              "| ISA denom: crude", d["crude_ratio_isa"], "age", pr(d["sir_isa"]), "| sex", {k: pr(v) for k, v in d["sex_period"].items() if k != "obs"})
        print("    by industry (rate, SIR age):", {k: (v["rate_per_1000"], v["sir_age"][0]) for k, v in R["status_industry"][s].items()})
        print("    by year ISA:", {y: (v["sir_age"][0], v["sir_isa"][0]) for y, v in d["by_year"].items()})
    print("check", R.get("check_statsmodels_all_pooled_age"))
