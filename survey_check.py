"""Compare the age structures estimated from residents with the 2025 Survey on Employment of Foreign Workers (MHLW;
establishment survey, Table 1, as of 30 Sep 2025; 10-year classes, %, age unknown excluded) and recompute the 2025 SIRs
with the survey structure (10-year totals from the survey, split within each class as in the main analysis)."""
import runpy, numpy as np, json
ns = runpy.run_path("analysis.py", run_name="x")
structure, expected_status, sir_ci = ns["structure"], ns["expected_status"], ns["sir_ci"]
SV = {"titp": [4.1, 70.9, 20.1, 3.0, 0.2, 0.0], "ssw": [0.3, 60.1, 34.3, 4.5, 0.5, 0.0],
      "engineer_humanities": [0.0, 35.7, 53.9, 8.8, 0.9, 0.4], "status_based": [1.2, 10.5, 22.6, 30.7, 22.5, 11.9]}
GR = [[0], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
out = {}
orig = structure
for s, sv in SV.items():
    sv = np.array(sv) / sum(sv)
    p = orig(2025, s); pa = p.sum(0)
    est10 = [pa[g].sum() * 100 for g in GR]
    q = p.copy()
    for g, t in zip(GR, sv):
        tot = p[:, g].sum()
        q[:, g] = p[:, g] / tot * t if tot > 0 else 0
    q = q / q.sum()
    def alt(y, status, weight=True, _s=s, _q=q):
        return _q if (status == _s and y == 2025) else orig(y, status, weight)
    expected_status.__globals__["structure"] = alt
    r_alt = expected_status(2025, s)
    expected_status.__globals__["structure"] = orig
    r_main = expected_status(2025, s)
    out[s] = dict(est10=[round(v, 1) for v in est10], survey10=[round(v * 100, 1) for v in sv], main=r_main, alt=r_alt)
    print(s, out[s]["est10"], out[s]["survey10"]); print("  main", r_main); print("  alt ", r_alt)
