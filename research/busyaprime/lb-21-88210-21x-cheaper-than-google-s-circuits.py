# CELL 1
import glob, hashlib, json, lzma, math, os, pathlib, platform, random, resource, shutil, subprocess, sys, time, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML, Markdown, display

T0 = time.monotonic()
START_UTC = time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())
# House style, spliced from engine/33_style.py (repo commit 7faa0ec3+local).
INK = "#1f2328"
MUTED = "#8b949e"
GRID = "#e6e8eb"
ACCENT = "#d9480f"
SECOND = "#1c7ed6"
PALETTE = [ACCENT, SECOND, "#495057", "#adb5bd", "#2b8a3e", "#862e9c"]
def apply() -> None:
    """Set matplotlib defaults. Safe to call more than once."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.figsize": (9, 4.6), "figure.dpi": 110, "savefig.dpi": 160,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.labelsize": 11, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "lines.linewidth": 2.0, "legend.frameon": False, "legend.fontsize": 10,
    })
def label_point(ax, x, y, text: str, color: str = ACCENT) -> None:
    """Print the number that matters on the chart, next to the point it describes."""
    ax.annotate(text, (x, y), xytext=(6, 6), textcoords="offset points", color=color,
                fontsize=11, fontweight="semibold")

REC = {'bundle': 'sub_012a', 'lb': '21.88210', 'lb_ref': '57709551', 'lb_date': '2026-10-27 17:59', 'score': 21.88210424920697, 'score_kind': "hosts' metric", 'zip_sha256': 'b28c1e4ff9e49b6a57f08c3ff0b0ba6ba42d5ee56a0282d3063d0eb643c6ce43', 'k_metric': 0, 'member_ks': [0, 1, 1, 1], 'rk': {'0': 1.1912604648115832, '1': 1.1967314991687559, '2': 1.2013880602795501, '3': 1.2055843238398525, '4': 1.209464049815962, '5': 1.2131052140444296, '6': 1.2165566016754124, '7': 1.2198515004832777, '8': 1.2230139975925918}, 'ref': {'invsqrt': {'cost': 2076774.233715282, 'q': 260, 'c': 7987.593206597238, 'shots': 200540928}, 'dqi': {'cost': 24439463583672.93, 'q': 1893, 't': 5725108.382109375, 'r': 1.1912604648115832, 'shots': 102400}, 'dfthcsos': {'cost': 3355599588.3155365, 'q': 730, 't': 93237.0, 'lam': 49.30136925057438, 'shots': 254464}}, 'expected': {'score': 21.88210424920697, 'costs': {'invsqrt': 103546.232446315, 'dqi': 442121458126.15326, 'dfthcsos': 355065066.0799457}, 'salts': {'invsqrt': 'ca571a29057a8d9e2042c151d67c5aa128e8c3b9', 'dqi': '4820d41f3b928dd8cdae7c09d773e86a3b072a89', 'dfthcsos': '43ec5f8405d34bc66694213a74aaec74bacf198a'}, 'measured': True, 'metric_record': 'data/qcc/runs/sub_012a/metric_local/metric_local.json', 'baseline': {'invsqrt': 2076774.233715282, 'dqi': 24439463583672.93, 'dfthcsos': 3355599588.3155365}}, 'member_ratio': {'invsqrt': 20.056491132030573, 'dqi': 55.08818287336645, 'dfthcsos': 9.450661044643567}, 'own_task': {'invsqrt': False, 'dqi': False, 'dfthcsos': True}, 'own_missing': [], 'own_lane_on': False, 'own_hist': {'bundle': 'sub_010f', 'lb': 17.53289, 'dqi': 31.694973085933167, 'dfthcsos': 8.86577302485725, 'invsqrt': 19.130133437753557}, 'chem3': [{'who': 'Ahmed Berat Ozer V9', 'lam': 47.55333620343766, 'cost': 355186805.3708027, 'e1': 0.014658039188274428, 'e2': 0.019997788521219226, 't': 17330.0, 'q': 431}, {'who': 'evgendvorkin', 'lam': 47.55236336608747, 'cost': 355179539.0248815, 'e1': 0.014656295260330143, 'e2': 0.01999778852121918, 't': 17330.0, 'q': 431}, {'who': 'busyaprime (this zip)', 'lam': 47.53703742955374, 'cost': 355065066.0799457, 'e1': 0.01999924529980602, 'e2': 0.019999499765582452, 't': 17330.0, 'q': 431}], 'relay': [{'label': '84 Q', 'who': 'Dmitrii Gluzdov', 'q': 84, 'c': 1854.7429, 'cost': 155798.4, 'n': 5, 'src': 'data/qcc/circuits/invsqrt_dmitrii84.verify.json'}, {'label': '82 Q EXP222', 'who': 'Phuc Thai', 'q': 82, 'c': 1716.0863, 'cost': 140719.077, 'n': 6, 'src': 'data/qcc/scout/regen/exp222/verify.json'}, {'label': '81 Q', 'who': 'busyaprime', 'q': 81, 'c': 1696.0866, 'cost': 137383.014, 'n': 6, 'src': 'data/qcc/circuits/invsqrt_e07b_q81anf.verify.json'}, {'label': '80 Q', 'who': 'busyaprime', 'q': 80, 'c': 1691.5865, 'cost': 135326.916, 'n': 6, 'src': 'data/qcc/circuits/invsqrt_e07b_q80.verify.json'}, {'label': '79 Q', 'who': 'busyaprime', 'q': 79, 'c': 1687.0866, 'cost': 133279.844, 'n': 2, 'src': 'data/qcc/circuits/invsqrt_c2d_q79f1m.full_unsalted.json'}, {'label': '78 Q', 'who': 'busyaprime', 'q': 78, 'c': 1697.0864, 'cost': 132372.742, 'n': 6, 'src': 'data/qcc/circuits/invsqrt_c2d_q78k.full_unsalted.json'}, {'label': '76 Q', 'who': 'mrmeows', 'q': 76, 'c': 1708.3369, 'cost': 129833.601, 'n': 6, 'src': 'data/qcc/circuits/invsqrt_h2_mrmeows.verify.json'}, {'label': '68 Q', 'who': 'mrmeows', 'q': 68, 'c': 1767.93, 'cost': 120219.243, 'n': 6, 'src': 'data/qcc/circuits/invsqrt_h3_mrmeows68.verify.json'}, {'label': '66 Q', 'who': 'mrmeows', 'q': 66, 'c': 1732.1805, 'cost': 114323.916, 'n': 6, 'src': 'data/qcc/circuits/invsqrt_h5_mrmeows66.verify.json'}, {'label': '63 Q', 'who': 'mrmeows', 'q': 63, 'c': 1723.1804, 'cost': 108560.363, 'n': 6, 'src': 'data/qcc/circuits/invsqrt_h7_mrmeows63.verify.json'}, {'label': '63 Q, newer', 'who': 'mrmeows', 'q': 63, 'c': 1643.5911, 'cost': 103546.239, 'n': 4, 'src': 'data/qcc/runs/h8_metric/ahmedberatozer__quantum-ai-df-thc-group-compression/metric_local.json'}], 'dqi_points': [{'label': 'q919', 'ours': True, 'q': 919, 't': 1418897.42, 'n': 11, 'src': 'data/qcc/circuits/dqi_mwc_proj70_q919.json'}, {'label': 'sigma pair', 'ours': True, 'q': 898, 't': 1056624.44, 'n': 6, 'src': 'data/qcc/circuits/dqi_e06_sigma.json'}, {'label': 'v7', 'ours': True, 'q': 870, 't': 1062679.71, 'n': 6, 'src': 'data/qcc/circuits/dqi_h1_v7.json'}, {'label': 'R5-A', 'ours': True, 'q': 872, 't': 942300.65, 'n': 6, 'src': 'data/qcc/circuits/dqi_r5a_cube_c.json'}, {'label': 'R9-A', 'ours': True, 'q': 870, 't': 939570.65, 'n': 6, 'src': 'data/qcc/circuits/dqi_r9a_f12.json'}, {'label': 'R12', 'ours': True, 'q': 870, 't': 935697.8, 'n': 12, 'src': 'data/qcc/circuits/dqi_r12_stack.json'}, {'label': 'R17', 'ours': True, 'q': 870, 't': 857617.63, 'n': 12, 'src': 'data/qcc/circuits/dqi_r17_q870ahm.json'}, {'label': 'R18', 'ours': True, 'q': 870, 't': 852940.6, 'n': 12, 'src': 'data/qcc/circuits/dqi_r18dqi_rml.json'}, {'label': 'SomeJoint gap', 'ours': False, 'q': 932, 't': 1424456.1, 'n': 6, 'src': 'data/qcc/circuits/dqi_sj_gap.json'}, {'label': 'DePie EXP-250a', 'ours': False, 'q': 873, 't': 1061302.65, 'n': 6, 'src': 'data/qcc/circuits/dqi_h1_depie_exp250a.json'}, {'label': 'ViberTeam 881', 'ours': False, 'q': 881, 't': 1116994.41, 'n': 6, 'src': 'data/qcc/circuits/dqi_h1_ana881.json'}, {'label': 'Ahmed V5', 'ours': False, 'q': 881, 't': 937970.2, 'n': 6, 'src': 'data/qcc/circuits/dqi_h5_ahmedv5.json'}, {'label': 'mrmeows Q880', 'ours': False, 'q': 880, 't': 905912.19, 'n': 17, 'src': 'data/qcc/circuits/dqi_h5_mrmq880.json'}, {'label': 'Ahmed V6', 'ours': False, 'q': 880, 't': 879488.19, 'n': 17, 'src': 'data/qcc/circuits/dqi_h6_ahmedv6.json'}, {'label': 'mrmeows Q870', 'ours': False, 'q': 870, 't': 884041.89, 'n': 17, 'src': 'data/qcc/circuits/dqi_h7_mrmq870.json'}, {'label': 'Ahmed V9 877 Q', 'ours': False, 'q': 877, 't': 482542.61, 'n': 4, 'src': 'data/qcc/work/dqi/r24/kernel/prof1/output/r24/verify_salt0.json', 'in_zip': True}], 'calib': [{'name': 'sub_001', 'ref': '57633891', 'date': '2026-10-25 14:57', 'lb': 9.73274, 'mean': 9.730041354425548, 'lo': 9.717893715543381, 'hi': 9.732750409942614, 'metric': 9.732748449666206}, {'name': 'sub_002', 'ref': '57636922', 'date': '2026-10-25 16:45', 'lb': 11.26131, 'mean': 11.258450107228343, 'lo': 11.244133133394417, 'hi': 11.261323825893772, 'metric': 11.261323366404415}, {'name': 'sub_003', 'ref': '57641376', 'date': '2026-10-25 19:51', 'lb': 11.53073, 'mean': 11.530731435857344, 'lo': 11.530729566315964, 'hi': 11.530732840867268, 'metric': 11.513131019493027}, {'name': 'sub_004', 'ref': '57644066', 'date': '2026-10-25 22:23', 'lb': 11.80113, 'mean': 11.810144569565017, 'lo': 11.801135500161424, 'hi': 11.819180913690968, 'metric': 11.801139032144413}, {'name': 'sub_005', 'ref': '57644937', 'date': '2026-10-25 23:19', 'lb': 11.94676, 'mean': 11.980526854277509, 'lo': 11.980524324117717, 'hi': 11.980529355698831, 'metric': 11.962243727520551}, {'name': 'sub_006', 'ref': '57648582', 'date': '2026-10-26 01:06', 'lb': 12.30713, 'mean': 12.289545455284356, 'lo': 12.258205472887044, 'hi': 12.307141923041922, 'metric': 12.307129479343239}, {'name': 'sub_007', 'ref': '57650567', 'date': '2026-10-26 02:29', 'lb': 13.91214, 'mean': 13.892263408879092, 'lo': 13.85683628162876, 'hi': 13.912154686879845, 'metric': 13.912140620359605}, {'name': 'sub_008', 'ref': '57655090', 'date': '2026-10-26 05:34', 'lb': 14.12017, 'mean': 14.153398375289367, 'lo': 14.120169466054321, 'hi': 14.16009659195503, 'metric': 14.160092220244987}, {'name': 'sub_009', 'ref': '57657370', 'date': '2026-10-26 06:51', 'lb': 15.37506, 'mean': 15.411240906907683, 'lo': 15.375058874033577, 'hi': 15.418534408294702, 'metric': 15.418529648061515}, {'name': 'sub_010f', 'ref': '57706707', 'date': '2026-10-27 16:05', 'lb': 17.53289, 'mean': 17.517615854714606, 'lo': 17.483506911852427, 'hi': 17.532926603005944, 'metric': 17.506164168532482}, {'name': 'sub_011a', 'ref': '57707884', 'date': '2026-10-27 16:54', 'lb': 21.8796, 'mean': 21.879603953380567, 'lo': 21.879603953380567, 'hi': 21.879603953380567, 'metric': 21.879603953380567}, {'name': 'sub_011b', 'ref': '57708355', 'date': '2026-10-27 17:12', 'lb': 21.87975, 'mean': 21.879752018812294, 'lo': 21.879752018812294, 'hi': 21.879752018812294, 'metric': None}], 'tide': [{'t': '2026-10-25T10:50:13', 'teams': 847, 'top1': 20.48324, 'gold': 14.49778, 'r100': 9.75359, 'median': 6.71415}, {'t': '2026-10-25T14:19:40', 'teams': 881, 'top1': 20.48324, 'gold': 14.49778, 'r100': 10.63866, 'median': 6.8053}, {'t': '2026-10-25T16:16:47', 'teams': 900, 'top1': 20.48324, 'gold': 14.61654, 'r100': 11.14623, 'median': 7.3248549999999994}, {'t': '2026-10-25T19:02:24', 'teams': 926, 'top1': 20.48324, 'gold': 15.50412, 'r100': 11.38437, 'median': 8.03407}, {'t': '2026-10-25T19:56:23', 'teams': 934, 'top1': 20.48324, 'gold': 15.50412, 'r100': 11.38437, 'median': 8.094195}, {'t': '2026-10-25T22:57:23', 'teams': 949, 'top1': 20.74782, 'gold': 16.48866, 'r100': 11.46076, 'median': 8.19645}, {'t': '2026-10-25T23:53:05', 'teams': 952, 'top1': 20.74782, 'gold': 16.53125, 'r100': 11.56806, 'median': 8.25824}, {'t': '2026-10-26T01:37:23', 'teams': 961, 'top1': 20.74782, 'gold': 16.53125, 'r100': 11.81693, 'median': 8.39401}, {'t': '2026-10-26T02:00:35', 'teams': 964, 'top1': 20.74782, 'gold': 16.53125, 'r100': 11.94383, 'median': 8.43645}, {'t': '2026-10-26T02:43:04', 'teams': 967, 'top1': 20.74782, 'gold': 16.53125, 'r100': 11.94713, 'median': 8.458}, {'t': '2026-10-26T02:57:16', 'teams': 970, 'top1': 21.09687, 'gold': 16.53125, 'r100': 11.97609, 'median': 8.49946}, {'t': '2026-10-26T03:42:39', 'teams': 976, 'top1': 21.09687, 'gold': 16.60291, 'r100': 12.267, 'median': 8.598875}, {'t': '2026-10-26T04:44:23', 'teams': 983, 'top1': 21.09687, 'gold': 16.8564, 'r100': 12.8722, 'median': 8.61985}, {'t': '2026-10-26T04:52:28', 'teams': 988, 'top1': 21.09687, 'gold': 16.8564, 'r100': 13.05507, 'median': 8.62436}, {'t': '2026-10-26T05:42:15', 'teams': 992, 'top1': 21.09687, 'gold': 17.04881, 'r100': 13.50478, 'median': 8.753499999999999}, {'t': '2026-10-26T06:03:40', 'teams': 994, 'top1': 21.48152, 'gold': 17.04881, 'r100': 13.50478, 'median': 8.95302}, {'t': '2026-10-26T06:45:21', 'teams': 998, 'top1': 21.48152, 'gold': 17.3061, 'r100': 13.71183, 'median': 8.96716}, {'t': '2026-10-26T07:18:45', 'teams': 1005, 'top1': 21.48152, 'gold': 17.3061, 'r100': 13.77415, 'median': 9.09001}, {'t': '2026-10-26T07:43:13', 'teams': 1010, 'top1': 21.48152, 'gold': 17.32026, 'r100': 13.92043, 'median': 9.114339999999999}, {'t': '2026-10-26T08:42:34', 'teams': 1022, 'top1': 21.48152, 'gold': 17.46401, 'r100': 13.95732, 'median': 9.13908}, {'t': '2026-10-26T09:42:30', 'teams': 1036, 'top1': 21.48152, 'gold': 17.76232, 'r100': 14.49721, 'median': 9.29737}, {'t': '2026-10-26T10:43:53', 'teams': 1047, 'top1': 22.3466, 'gold': 17.97652, 'r100': 14.6985, 'median': 9.35916}, {'t': '2026-10-26T11:42:18', 'teams': 1060, 'top1': 22.3466, 'gold': 18.7757, 'r100': 15.08992, 'median': 9.3665}, {'t': '2026-10-26T12:42:19', 'teams': 1075, 'top1': 22.52237, 'gold': 18.7757, 'r100': 15.13265, 'median': 9.38293}, {'t': '2026-10-26T13:42:45', 'teams': 1085, 'top1': 22.886, 'gold': 18.7757, 'r100': 15.21543, 'median': 9.49654}, {'t': '2026-10-26T14:42:40', 'teams': 1093, 'top1': 22.886, 'gold': 19.15996, 'r100': 15.57511, 'median': 9.55447}, {'t': '2026-10-26T15:44:03', 'teams': 1101, 'top1': 23.35228, 'gold': 19.52616, 'r100': 15.74971, 'median': 9.72508}, {'t': '2026-10-26T16:43:30', 'teams': 1111, 'top1': 23.35228, 'gold': 19.67522, 'r100': 15.91387, 'median': 10.01668}, {'t': '2026-10-27T16:09:08', 'teams': 1271, 'top1': 24.52926, 'gold': 22.69946, 'r100': 21.8796, 'median': 14.34435}, {'t': '2026-10-27T16:35:36', 'teams': 1274, 'top1': 24.52926, 'gold': 22.69946, 'r100': 21.8796, 'median': 14.42078}, {'t': '2026-10-27T17:58:48', 'teams': 1282, 'top1': 24.52926, 'gold': 22.72031, 'r100': 21.8796, 'median': 14.49721}, {'t': '2026-10-27T18:31:11', 'teams': 1290, 'top1': 24.52926, 'gold': 22.72031, 'r100': 21.8796, 'median': 14.49721}], 'tide_gaps': [('2026-10-26T16:43:30', '2026-10-27T16:09:08')], 'ours_steps': [{'t': '2026-10-25T14:57:01', 'score': 9.73274, 'name': 'sub_001'}, {'t': '2026-10-25T16:45:26', 'score': 11.26131, 'name': 'sub_002'}, {'t': '2026-10-25T19:51:03', 'score': 11.53073, 'name': 'sub_003'}, {'t': '2026-10-25T22:23:20', 'score': 11.80113, 'name': 'sub_004'}, {'t': '2026-10-25T23:19:08', 'score': 11.94676, 'name': 'sub_005'}, {'t': '2026-10-26T01:06:30', 'score': 12.30713, 'name': 'sub_006'}, {'t': '2026-10-26T02:29:23', 'score': 13.91214, 'name': 'sub_007'}, {'t': '2026-10-26T05:34:33', 'score': 14.12017, 'name': 'sub_008'}, {'t': '2026-10-26T06:51:08', 'score': 15.37506, 'name': 'sub_009'}, {'t': '2026-10-27T16:05:14', 'score': 17.53289, 'name': 'sub_010f'}, {'t': '2026-10-27T16:54:25', 'score': 21.8796, 'name': 'sub_011a'}, {'t': '2026-10-27T17:12:58', 'score': 21.87975, 'name': 'sub_011b'}, {'t': '2026-10-27T17:59:01', 'score': 21.8821, 'name': 'sub_012a'}], 'board_scores': [24.52926, 24.03231, 23.79988, 23.6686, 23.48356, 23.4449, 23.4339, 22.97273, 22.94242, 22.86405, 22.83577, 22.72031, 22.69946, 22.40372, 22.37308, 22.36808, 22.3193, 22.31763, 22.24034, 22.13397, 22.12322, 22.11343, 22.082, 22.08007, 22.04459, 22.04217, 22.03817, 22.03064, 22.01699, 22.00475, 22.00433, 21.9963, 21.99538, 21.99204, 21.99184, 21.99184, 21.98808, 21.98505, 21.98374, 21.97897, 21.97897, 21.96955, 21.96752, 21.96613, 21.95244, 21.94794, 21.94367, 21.94335, 21.93489, 21.9333, 21.93258, 21.92735, 21.92694, 21.92609, 21.92045, 21.91718, 21.91394, 21.91388, 21.91252, 21.91099, 21.90745, 21.90648, 21.90446, 21.90406, 21.90153, 21.89963, 21.89747, 21.8974, 21.89396, 21.89152, 21.88717, 21.88322, 21.8827, 21.88256, 21.8821, 21.88089, 21.88044, 21.8798, 21.87978, 21.87976, 21.87976, 21.87975, 21.87975, 21.87975, 21.87975, 21.87966, 21.87966, 21.87966, 21.87963, 21.87961, 21.87961, 21.87961, 21.87961, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.8796, 21.87735, 21.87693, 21.87073, 21.85292, 21.82541, 21.81933, 21.80703, 21.80703, 21.80703, 21.80703, 21.80381, 21.80337, 21.8025, 21.78241, 21.78159, 21.76799, 21.76785, 21.76633, 21.76364, 21.7569, 21.7452, 21.74454, 21.7443, 21.74094, 21.71925, 21.7106, 21.69212, 21.6889, 21.68875, 21.68875, 21.68875, 21.68875, 21.68875, 21.68875, 21.68086, 21.67197, 21.66814, 21.65879, 21.64936, 21.64516, 21.63518, 21.63285, 21.59972, 21.59881, 21.58801, 21.5854, 21.58209, 21.57964, 21.57928, 21.57157, 21.56747, 21.56653, 21.56604, 21.56592, 21.5646, 21.56438, 21.56301, 21.55944, 21.55792, 21.55744, 21.55675, 21.55572, 21.55572, 21.55572, 21.55383, 21.55383, 21.55383, 21.55383, 21.55339, 21.54763, 21.54688, 21.54317, 21.54244, 21.53452, 21.53452, 21.53293, 21.5315, 21.53142, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.53135, 21.48341, 21.48158, 21.47224, 21.46942, 21.46791, 21.43735, 21.43735, 21.43735, 21.43735, 21.43735, 21.43735, 21.43735, 21.43735, 21.43735, 21.43735, 21.43735, 21.43735, 21.43735, 21.43735, 21.43735, 21.43735, 21.43735, 21.43015, 21.42915, 21.42865, 21.42389, 21.41815, 21.40999, 21.40999, 21.40999, 21.40999, 21.40999, 21.40279, 21.39816, 21.39434, 21.39434, 21.39434, 21.39434, 21.39434, 21.39434, 21.39434, 21.39434, 21.39434, 21.39434, 21.39434, 21.39434, 21.38909, 21.38884, 21.38884, 21.38877, 21.38877, 21.38877, 21.38877, 21.38877, 21.38877, 21.38877, 21.38877, 21.38877, 21.38877, 21.38877, 21.38877, 21.38877, 21.38877, 21.38198, 21.37767, 21.37714, 21.37712, 21.37712, 21.37712, 21.37712, 21.37712, 21.37712, 21.37712, 21.37712, 21.37712, 21.37712, 21.37712, 21.37712, 21.37712, 21.37712, 21.37712, 21.37712, 21.37712, 21.37712, 21.37712, 21.37712, 21.37712, 21.36637, 21.36187, 21.34628, 21.33109, 21.32981, 21.29579, 21.29216, 21.28379, 21.28379, 21.28379, 21.28379, 21.28379, 21.26342, 21.22616, 21.20043, 21.19299, 21.19187, 21.17177, 21.1692, 21.16823, 21.16807, 21.16807, 21.16807, 21.16807, 21.16807, 21.16807, 21.16807, 21.16807, 21.16807, 21.16807, 21.16807, 21.16807, 21.16807, 21.16807, 21.16807, 21.16807, 21.15049, 21.14441, 21.14441, 21.14441, 21.14441, 21.14422, 21.14106, 21.14106, 21.13249, 21.13249, 21.1277, 21.12064, 21.09512, 21.05785, 21.05089, 21.04525, 21.02529, 21.02524, 21.02523, 21.02523, 21.02523, 21.02523, 21.02523, 21.02523, 21.02523, 21.02523, 21.02523, 21.02523, 21.02523, 21.02523, 21.02523, 21.02523, 21.02523, 21.02523, 20.98527, 20.98527, 20.98527, 20.98527, 20.98527, 20.98527, 20.98527, 20.98527, 20.98527, 20.98527, 20.98527, 20.98527, 20.93277, 20.88377, 20.8786, 20.87605, 20.83637, 20.75901, 20.73006, 20.70983, 20.70081, 20.68963, 20.67047, 20.67047, 20.67047, 20.67047, 20.67047, 20.67047, 20.67047, 20.67047, 20.67047, 20.64104, 20.57453, 20.50687, 20.50608, 20.50608, 20.50608, 20.50608, 20.50608, 20.50608, 20.46785, 20.46711, 20.46711, 20.46711, 20.46711, 20.46711, 20.46711, 20.46711, 20.46711, 20.46711, 20.38703, 20.38004, 20.37251, 20.23673, 20.18843, 20.16956, 20.13369, 20.09834, 20.04438, 20.03884, 19.9418, 19.94179, 19.81709, 19.78441, 19.77353, 19.69995, 19.62592, 19.62212, 19.62212, 19.57899, 19.57899, 19.57899, 19.57896, 19.57896, 19.57896, 19.57896, 19.57896, 19.57896, 19.55036, 19.49787, 19.49787, 19.49785, 19.49785, 19.49785, 19.49785, 19.34265, 19.31178, 19.27697, 19.25494, 19.24359, 19.23822, 19.21488, 19.21488, 19.21488, 19.21488, 19.15044, 18.70535, 18.55984, 18.41307, 18.41307, 18.41307, 18.41305, 18.41305, 18.41305, 18.41305, 18.41305, 18.41305, 18.41305, 18.33292, 18.33292, 18.33292, 18.33292, 18.33292, 18.33292, 18.33292, 18.33292, 18.25295, 18.06011, 17.99064, 17.84868, 17.78942, 17.76232, 17.46401, 17.45461, 17.30179, 17.21187, 17.20417, 17.14234, 16.90171, 16.90171, 16.86827, 16.80182, 16.68693, 16.66684, 16.29982, 16.21622, 16.18497, 16.14666, 16.13441, 15.99994, 15.93682, 15.93415, 15.90142, 15.88401, 15.82593, 15.7962, 15.74971, 15.74971, 15.74971, 15.74971, 15.74971, 15.74971, 15.74971, 15.74971, 15.74971, 15.74971, 15.74971, 15.74971, 15.74971, 15.70929, 15.69537, 15.68727, 15.68046, 15.55024, 15.50238, 15.4544, 15.33759, 15.33759, 15.33759, 15.33759, 15.2551, 15.17926, 15.14751, 15.14394, 15.14217, 15.13265, 15.13265, 15.13265, 15.13265, 15.08992, 15.08992, 15.08992, 15.08992, 15.08992, 15.08992, 15.08992, 15.08992, 15.08992, 15.08992, 15.08256, 15.06689, 14.99359, 14.90713, 14.88842, 14.76733, 14.70571, 14.67765, 14.67668, 14.57387, 14.50108, 14.49721, 14.49721, 14.49721, 14.49721, 14.49721, 14.49721, 14.49721, 14.49721, 14.49721, 14.49721, 14.34435, 14.02829, 13.95732, 13.92043, 13.92043, 13.92043, 13.83719, 13.83357, 13.77872, 13.74828, 13.56192, 13.54124, 13.50478, 13.50478, 13.50478, 13.50478, 13.50478, 13.50478, 13.50478, 13.50478, 13.50478, 13.50478, 13.50478, 13.45869, 13.34231, 13.28394, 12.8722, 12.86485, 12.83416, 12.7788, 12.64809, 12.51715, 12.33542, 12.28805, 12.26251, 12.18239, 12.15671, 12.03025, 11.97107, 11.95949, 11.94676, 11.94676, 11.85856, 11.85043, 11.81368, 11.81368, 11.78129, 11.67289, 11.56806, 11.56475, 11.53736, 11.43958, 11.43958, 11.43958, 11.42796, 11.41171, 11.38437, 11.38437, 11.38437, 11.38437, 11.38437, 11.38437, 11.38437, 11.32827, 11.32827, 11.31762, 11.3146, 11.29773, 11.29773, 11.29773, 11.29773, 11.2829, 11.2829, 11.28267, 11.27969, 11.27369, 11.2242, 11.2242, 11.2242, 11.2242, 11.17555, 11.12982, 11.04919, 11.04376, 10.95803, 10.94951, 10.92347, 10.91018, 10.88149, 10.88057, 10.88048, 10.88048, 10.88048, 10.88048, 10.83, 10.6466, 10.64388, 10.61949, 10.61612, 10.58328, 10.55511, 10.5461, 10.5461, 10.5461, 10.5461, 10.52187, 10.52187, 10.52187, 10.43616, 10.40495, 10.34781, 10.31323, 10.263, 10.22567, 10.21976, 10.20097, 10.14078, 10.09155, 10.0418, 10.01668, 9.79066, 9.76008, 9.68912, 9.60956, 9.60884, 9.58065, 9.55447, 9.55115, 9.52885, 9.51983, 9.51983, 9.51483, 9.50889, 9.47596, 9.46478, 9.44792, 9.38293, 9.38293, 9.37374, 9.37071, 9.36842, 9.36801, 9.36499, 9.35916, 9.31576, 9.31449, 9.31334, 9.30235, 9.30235, 9.3002, 9.29239, 9.26724, 9.19604, 9.13908, 9.13908, 9.13908, 9.12796, 9.10493, 9.08981, 9.05124, 8.95302, 8.95302, 8.95302, 8.95302, 8.93469, 8.76281, 8.74419, 8.62436, 8.61985, 8.61985, 8.59832, 8.55186, 8.49219, 8.4598, 8.458, 8.44405, 8.43645, 8.43645, 8.40876, 8.33337, 8.21615, 8.20796, 8.19645, 8.16543, 8.10563, 8.04343, 8.03884, 8.03504, 8.0331, 8.03297, 8.0136, 8.00454, 7.97482, 7.97277, 7.93016, 7.91014, 7.906, 7.8246, 7.77301, 7.72915, 7.65967, 7.63096, 7.61577, 7.61508, 7.5784, 7.57135, 7.56759, 7.52178, 7.33869, 7.28317, 7.25527, 7.2528, 7.11435, 7.02142, 7.014, 7.00207, 6.97354, 6.80762, 6.80621, 6.8053, 6.79051, 6.75582, 6.74471, 6.74471, 6.72748, 6.72238, 6.71534, 6.71415, 6.70503, 6.70503, 6.70503, 6.70503, 6.689, 6.68886, 6.68108, 6.68108, 6.68108, 6.68108, 6.64525, 6.64433, 6.64383, 6.61767, 6.61026, 6.564, 6.55188, 6.54255, 6.53689, 6.52837, 6.51066, 6.46352, 6.45055, 6.41119, 6.40844, 6.39943, 6.39471, 6.37885, 6.37294, 6.37124, 6.36732, 6.351, 6.351, 6.351, 6.351, 6.32784, 6.28527, 6.28129, 6.25697, 6.25697, 6.25688, 6.22965, 6.22116, 6.20972, 6.20313, 6.19667, 6.17788, 6.13796, 6.13549, 6.12847, 6.10918, 6.09407, 6.06344, 5.95834, 5.94297, 5.8232, 5.78784, 5.78629, 5.78629, 5.77246, 5.77242, 5.77242, 5.77242, 5.7603, 5.75763, 5.75601, 5.7325, 5.72245, 5.71594, 5.71594, 5.71594, 5.70521, 5.70127, 5.69358, 5.68564, 5.68551, 5.6838, 5.67664, 5.67079, 5.67004, 5.66339, 5.63791, 5.58663, 5.54184, 5.48064, 5.4803, 5.46279, 5.40859, 5.40266, 5.26613, 5.13369, 5.01596, 5.00363, 4.96916, 4.88354, 4.84971, 4.81697, 4.76247, 4.74284, 4.72842, 4.60344, 4.5309, 4.46205, 4.40002, 4.37399, 4.3641, 4.25631, 4.24604, 4.21329, 4.19375, 4.15429, 4.13422, 4.1294, 4.1294, 4.1294, 4.1294, 4.03978, 4.03428, 3.90894, 3.90426, 3.90426, 3.90426, 3.90307, 3.90307, 3.9013, 3.90011, 3.85524, 3.8289, 3.80761, 3.7967, 3.7138, 3.68891, 3.67884, 3.66363, 3.64818, 3.63195, 3.62271, 3.52051, 3.26938, 3.21579, 3.13944, 3.10611, 3.10141, 2.81476, 2.77339, 2.72662, 2.59886, 2.53419, 2.51961, 2.50941, 2.46837, 2.46441, 2.43167, 2.40987, 2.40231, 2.36122, 2.34179, 2.33999, 2.32601, 2.3242, 2.28984, 2.2897, 2.27645, 2.26491, 2.2563, 2.24885, 2.24193, 2.21305, 2.181, 2.16213, 2.14839, 2.14548, 2.08382, 2.08074, 2.07108, 2.06965, 2.05073, 2.01765, 2.00897, 1.98254, 1.97639, 1.96719, 1.96318, 1.96207, 1.94052, 1.92568, 1.86639, 1.85826, 1.84341, 1.84225, 1.83833, 1.83435, 1.83225, 1.82066, 1.81703, 1.81524, 1.78857, 1.74592, 1.73724, 1.73564, 1.71756, 1.69352, 1.69347, 1.67825, 1.6695, 1.65452, 1.61532, 1.61396, 1.60303, 1.59613, 1.59555, 1.59543, 1.59422, 1.58598, 1.57692, 1.56916, 1.56802, 1.56708, 1.5658, 1.54586, 1.51831, 1.51624, 1.51008, 1.49725, 1.48264, 1.46831, 1.43269, 1.40438, 1.37489, 1.37391, 1.35027, 1.34199, 1.33485, 1.33102, 1.31744, 1.30737, 1.30488, 1.26826, 1.2065, 1.2031, 1.19842, 1.19021, 1.16694, 1.16626, 1.1589, 1.15849, 1.15758, 1.15562, 1.14985, 1.13367, 1.12146, 1.11487, 1.11433, 1.112, 1.10553, 1.09357, 1.08972, 1.08027, 1.06681, 1.05676, 1.05632, 1.05201, 1.05181, 1.05164, 1.04789, 1.04613, 1.04437, 1.04057, 1.03394, 1.03348, 1.03281, 1.03146, 1.02982, 1.02903, 1.02774, 1.0265, 1.02548, 1.0244, 1.02289, 1.02283, 1.02274, 1.02274, 1.02274, 1.02274, 1.02274, 1.02274, 1.02274, 1.02274, 1.02274, 1.02274, 1.02274, 1.02274, 1.02274, 1.02274, 1.02274, 1.02274, 1.02194, 1.02194, 1.02194, 1.01749, 1.01749, 1.01749, 1.01749, 1.01749, 1.0169, 1.01553, 1.01444, 1.01444, 1.01444, 1.01444, 1.01444, 1.01364, 1.01247, 1.01232, 1.01094, 1.01002, 1.00996, 1.00996, 1.00949, 1.00886, 1.00808, 1.00475, 1.00475, 1.00212, 1.00084, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99999, 0.99777, 0.98294, 0.97849, 0.97849, 0.97849, 0.97849, 0.97849, 0.97849], 'clusters': [{'score': 21.8796, 'n': 82, 'notebook': 'ahmedberatozer/quantum-ai-df-thc-group-compression'}, {'score': 21.55383, 'n': 4, 'notebook': 'leoprovorov/the-unbearable-lightness-of-quantum-circuits'}, {'score': 21.53135, 'n': 32, 'notebook': 'saddamaditya/narrowed-877q-dqi-and-progressive-r10-chemistry'}, {'score': 21.39434, 'n': 12, 'notebook': 'alexchilton/21-39-lb-verified-three-task-portfolio'}, {'score': 21.38877, 'n': 14, 'notebook': 'heliosli/qcc-j5-chemistry-loaders-deeper-lookup-plans'}, {'score': 21.37712, 'n': 21, 'notebook': 'somejoint/quantum-circuits-21-34-local-gm-fused-dqi'}, {'score': 21.28379, 'n': 5, 'notebook': 'dmitriigluzdov/quantum-circuit-compilation-from-source'}, {'score': 21.02523, 'n': 16, 'notebook': 'mrmeows/21-0-circuits-let-s-all-reach-the-plateau'}, {'score': 20.67047, 'n': 9, 'notebook': 'manwithacat/composite-v3-mrmeows-somejoint'}, {'score': 19.57896, 'n': 6, 'notebook': 'lightningv08/bezout-schur-5-word-qrom'}, {'score': 19.49785, 'n': 4, 'notebook': 'avazaku/quantum-circuits-7-bit-bounds'}, {'score': 19.21488, 'n': 4, 'notebook': 'manwithacat/how-the-public-19-3-portfolio-works'}], 'field': [{'ref': 'ahmedberatozer/quantum-ai-df-thc-group-compression', 'page': 'https://www.kaggle.com/code/ahmedberatozer/quantum-ai-df-thc-group-compression', 'kind': 'notebook', 'version': 'V9', 'score': '21.87960', 'votes': '91', 'licence': 'Apache 2.0', 'rerun': 21.8796, 'ratios': {'invsqrt': 20.06, 'dqi': 55.28, 'dfthcsos': 9.45}}, {'ref': 'leoprovorov/the-unbearable-lightness-of-quantum-circuits', 'page': 'https://www.kaggle.com/code/leoprovorov/the-unbearable-lightness-of-quantum-circuits', 'kind': 'notebook', 'version': 'V25', 'score': '21.55383', 'votes': '21', 'licence': 'Apache 2.0', 'rerun': 21.55384, 'ratios': {'invsqrt': 20.06, 'dqi': 54.73, 'dfthcsos': 9.12}}, {'ref': 'kozykappa/lb-21-38884-quantum-ai-q-circuits', 'page': 'https://www.kaggle.com/code/kozykappa/lb-21-38884-quantum-ai-q-circuits', 'kind': 'notebook', 'version': 'V12', 'score': '21.38884', 'votes': '19', 'licence': 'Apache 2.0', 'rerun': None, 'ratios': None}, {'ref': 'heliosli/qcc-j5-chemistry-loaders-deeper-lookup-plans', 'page': 'https://www.kaggle.com/code/heliosli/qcc-j5-chemistry-loaders-deeper-lookup-plans', 'kind': 'notebook', 'version': 'V7', 'score': '21.38877', 'votes': '12', 'licence': 'Apache 2.0', 'rerun': None, 'ratios': None}, {'ref': 'somejoint/quantum-circuits-21-34-local-gm-fused-dqi', 'page': 'https://www.kaggle.com/code/somejoint/quantum-circuits-21-34-local-gm-fused-dqi', 'kind': 'notebook', 'version': 'V12', 'score': '21.37712', 'votes': '60', 'licence': 'Apache 2.0', 'rerun': None, 'ratios': None}, {'ref': 'dmitriigluzdov/quantum-circuit-compilation-from-source', 'page': 'https://www.kaggle.com/code/dmitriigluzdov/quantum-circuit-compilation-from-source', 'kind': 'notebook', 'version': 'V24', 'score': '21.28379', 'votes': '73', 'licence': 'Apache 2.0', 'rerun': None, 'ratios': None}, {'ref': 'zongaobian/quantum-circuits-signed-digits-bm-psi-qrom', 'page': 'https://www.kaggle.com/code/zongaobian/quantum-circuits-signed-digits-bm-psi-qrom', 'kind': 'notebook', 'version': 'V5', 'score': '21.08319', 'votes': '7', 'licence': 'Apache 2.0', 'rerun': None, 'ratios': None}, {'ref': 'mrmeows/21-0-circuits-let-s-all-reach-the-plateau', 'page': 'https://www.kaggle.com/code/mrmeows/21-0-circuits-let-s-all-reach-the-plateau', 'kind': 'notebook', 'version': 'V8', 'score': '21.02523', 'votes': '43', 'licence': 'Apache 2.0', 'rerun': None, 'ratios': None}, {'ref': 'lightningv08/bezout-schur-5-word-qrom', 'page': 'https://www.kaggle.com/code/lightningv08/bezout-schur-5-word-qrom', 'kind': 'notebook', 'version': 'V15', 'score': '19.57896', 'votes': '52', 'licence': 'Apache 2.0', 'rerun': None, 'ratios': None}, {'ref': 'saddamaditya/narrowed-877q-dqi-and-progressive-r10-chemistry', 'page': 'https://www.kaggle.com/code/saddamaditya/narrowed-877q-dqi-and-progressive-r10-chemistry', 'kind': 'notebook', 'version': 'V2', 'score': 'none shown', 'votes': '6', 'licence': 'Apache 2.0', 'rerun': None, 'ratios': None}, {'ref': 'prvsiyan/verifier-guided-quantum-circuit-optimization', 'page': 'https://www.kaggle.com/code/prvsiyan/verifier-guided-quantum-circuit-optimization', 'kind': 'notebook', 'version': 'V14', 'score': 'none shown', 'votes': '7', 'licence': 'Apache 2.0', 'rerun': None, 'ratios': None}, {'ref': 'alexchilton/21-39-lb-verified-three-task-portfolio', 'page': 'https://www.kaggle.com/code/alexchilton/21-39-lb-verified-three-task-portfolio', 'kind': 'notebook', 'version': 'V46', 'score': 'none shown', 'votes': '38', 'licence': 'Apache 2.0', 'rerun': None, 'ratios': None}, {'ref': 'leoprovorov/but-some-q-squared-width-lab', 'page': 'https://www.kaggle.com/code/leoprovorov/but-some-q-squared-width-lab', 'kind': 'notebook', 'version': 'V22', 'score': 'none shown', 'votes': '12', 'licence': 'Apache 2.0', 'rerun': None, 'ratios': None}, {'ref': 'ahmedberatozer/vqcc-v9-components', 'page': 'https://www.kaggle.com/datasets/ahmedberatozer/vqcc-v9-components', 'kind': 'dataset', 'version': 'V1 (20.77 MB)', 'score': '-', 'votes': '2', 'licence': 'Other (specified in description): "Our own parts are released under Apache-2.0"', 'rerun': None, 'ratios': None}, {'ref': 'mrmeows/qcc-1375-open-circuits', 'page': 'https://www.kaggle.com/datasets/mrmeows/qcc-1375-open-circuits', 'kind': 'dataset', 'version': 'V8 (1.17 GB)', 'score': '-', 'votes': '5', 'licence': 'Apache 2.0', 'rerun': None, 'ratios': None}, {'ref': 'zongaobian/quantum-circuits-signed-digits-bmpsi-and-qrom', 'page': 'https://www.kaggle.com/datasets/zongaobian/quantum-circuits-signed-digits-bmpsi-and-qrom', 'kind': 'dataset', 'version': 'V3 (68.35 MB)', 'score': '-', 'votes': '-', 'licence': 'Other (specified in description)', 'rerun': None, 'ratios': None}], 'ledger': {'invsqrt': {'rows': 108, 'failed': 50, 'other': 58, 'days': ['2026-10-25', '2026-10-26', '2026-10-27']}, 'dqi': {'rows': 84, 'failed': 21, 'other': 63, 'days': ['2026-10-25', '2026-10-26', '2026-10-27']}, 'dfthcsos': {'rows': 212, 'failed': 30, 'other': 182, 'days': ['2026-10-25', '2026-10-26', '2026-10-27']}}, 'dead': {'invsqrt': ['merging the high and low phase passes; keeping the curvature register alive', 'one phase pass in any register schedule (H kept, H recomputed at the pass, in-place Horner, per-row phasing): best case 121,404 at Q 78 only if H were free; H costs >= 7 Q'], 'dqi': ['mask projection beyond 70 positions (exhausted)', 'stored Psi in the Forney scan (needs 34 words); one inverse per block of scan positions', "port of uncompute's zlin 71-point trick (already in the member: 70-position projection + per-shot skip of zero residuals)"], 'dfthcsos': ['Gram-preserving rotation of the copy index (no gain)', 'shrinking (1,496,384) to fewer vectors (B 480, 460) or columns (C 360 and below) in the V11 family', 'a rounding-noise-optimal Givens tree (shared tree: noise x0.978)']}, 'distinct': {'invsqrt': 2, 'dqi': 6, 'dfthcsos': 9}, 'n_zips': 14}  # every recorded number the figures use (written by build.py from the repo's record files)
PINS = {'invsqrt.kmb': {'sha256': '0766a111a76f241aa57fbbed466436d743be13436943ddb0628c291ca07d67d6', 'source': 'busyaprime/33-qcc-v8-members', 'file': 'invsqrt.kmb', 'codec': None, 'member': None, 'owner': 'busyaprime'}, 'dqi.kmb': {'sha256': 'f3d335c3f693aaab93bcea8750605f206dd2ba6d04ad5bf9a6de1de1ac5ec3fb', 'source': 'busyaprime/33-qcc-v8-members', 'file': 'dqi.kmb', 'codec': None, 'member': None, 'owner': 'busyaprime'}, 'dfthcsos/angle_k_0.kmb': {'sha256': '0942f5305a87c5a4fcf6651e62df9101cec4671415f169a71de60ba9ea75ff7c', 'source': 'busyaprime/33-qcc-v8-members', 'file': 'dfthcsos__angle_k_0.kmb', 'codec': None, 'member': None, 'owner': 'busyaprime'}, 'dfthcsos/angle_k_1.kmb': {'sha256': '54951c2665fac7bf59aac7c4889f6b3785fe080951b9914168d86a498cee7b49', 'source': 'busyaprime/33-qcc-v8-members', 'file': 'dfthcsos__angle_k_1.kmb', 'codec': None, 'member': None, 'owner': 'busyaprime'}, 'dfthcsos/angle_k_10.kmb': {'sha256': '2c7006937399980a5d4642acbbd7033dbc1517eca941cd84eb86c752c66c6120', 'source': 'busyaprime/33-qcc-v8-members', 'file': 'dfthcsos__angle_k_10.kmb', 'codec': None, 'member': None, 'owner': 'busyaprime'}, 'dfthcsos/angle_k_11.kmb': {'sha256': 'd4342edd711b9bea1afd9ef79d931bfbd1845078b45b65648273fb033d225f50', 'source': 'busyaprime/33-qcc-v8-members', 'file': 'dfthcsos__angle_k_11.kmb', 'codec': None, 'member': None, 'owner': 'busyaprime'}, 'dfthcsos/angle_k_12.kmb': {'sha256': '785565ad2fbd99b53b89931795ceb9c05e7b1524c7f1a7f58ffbf6baa5e0d8b5', 'source': 'busyaprime/33-qcc-v8-members', 'file': 'dfthcsos__angle_k_12.kmb', 'codec': None, 'member': None, 'owner': 'busyaprime'}, 'dfthcsos/angle_k_13.kmb': {'sha256': 'c265d77603a919b83c1d20ced3252692a5f902b0175c6cfdbac7ce584bd1f9c2', 'source': 'busyaprime/33-qcc-v8-members', 'file': 'dfthcsos__angle_k_13.kmb', 'codec': None, 'member': None, 'owner': 'busyaprime'}, 'dfthcsos/angle_k_14.kmb': {'sha256': 'c9df4f1975b63ed9014ecffe693337c6d6919d81a7b32025e78dbdc664486881', 'source': 'busyaprime/33-qcc-v8-members', 'file': 'dfthcsos__angle_k_14.kmb', 'codec': None, 'member': None, 'owner': 'busyaprime'}, 'dfthcsos/angle_k_2.kmb': {'sha256': 'c618f287eb496c1845b7200b35f0fdd0491f1184f3b09f118a7d421234c8a14d', 'source': 'busyaprime/33-qcc-v8-members', 'file': 'dfthcsos__angle_k_2.kmb', 'codec': None, 'member': None, 'owner': 'busyaprime'}, 'dfthcsos/angle_k_3.kmb': {'sha256': 'f091a367d27bdf2a721122ec3515784a7d0513c869e7b0c452db147798197599', 'source': 'busyaprime/33-qcc-v8-members', 'file': 'dfthcsos__angle_k_3.kmb', 'codec': None, 'member': None, 'owner': 'busyaprime'}, 'dfthcsos/angle_k_4.kmb': {'sha256': '945ad68ef9be3e6c4f3e30562ffe96399af8e51ade7877e3e9dc9d8b870154b5', 'source': 'busyaprime/33-qcc-v8-members', 'file': 'dfthcsos__angle_k_4.kmb', 'codec': None, 'member': None, 'owner': 'busyaprime'}, 'dfthcsos/angle_k_5.kmb': {'sha256': '47a56f9bf5c95b0e2006b390ba868f874d439c594d16abc729d19c2b1d894928', 'source': 'busyaprime/33-qcc-v8-members', 'file': 'dfthcsos__angle_k_5.kmb', 'codec': None, 'member': None, 'owner': 'busyaprime'}, 'dfthcsos/angle_k_6.kmb': {'sha256': '85f4675c9b853e8714aff1af5448b4e8321a5cdf6087d0105ee57c3bdcc306d1', 'source': 'busyaprime/33-qcc-v8-members', 'file': 'dfthcsos__angle_k_6.kmb', 'codec': None, 'member': None, 'owner': 'busyaprime'}, 'dfthcsos/angle_k_7.kmb': {'sha256': '1946e4372780fbb04da292d3544d206a6676b72a38546c461df813761e67225b', 'source': 'busyaprime/33-qcc-v8-members', 'file': 'dfthcsos__angle_k_7.kmb', 'codec': None, 'member': None, 'owner': 'busyaprime'}, 'dfthcsos/angle_k_8.kmb': {'sha256': '95c635076a3e387d967715bf6fc203466aaf0de654cef8bc550892f7ba7ca35e', 'source': 'busyaprime/33-qcc-v8-members', 'file': 'dfthcsos__angle_k_8.kmb', 'codec': None, 'member': None, 'owner': 'busyaprime'}, 'dfthcsos/angle_k_9.kmb': {'sha256': '7a8338e7ef8a04718c07998539b8d536245bf38e1844599a13219a45f57097b4', 'source': 'busyaprime/33-qcc-v8-members', 'file': 'dfthcsos__angle_k_9.kmb', 'codec': None, 'member': None, 'owner': 'busyaprime'}, 'dfthcsos/submission.npz': {'sha256': 'a60e21cb380474976b7d99dbf97889beb0784928d3a6c0becef12bc982496ab6', 'source': 'busyaprime/33-qcc-v8-members', 'file': 'dfthcsos__submission.npz', 'codec': None, 'member': None, 'owner': 'busyaprime'}}  # archive name: where the bytes live and their SHA-256
OWN_PINS = {}  # our own best message-repair and chemistry members
ZIPMETA = {'dfthcsos/angle_k_0.kmb': {'date_time': [2026, 10, 27, 10, 45, 58], 'mode': 33206}, 'dfthcsos/angle_k_1.kmb': {'date_time': [2026, 10, 27, 10, 45, 58], 'mode': 33206}, 'dfthcsos/angle_k_10.kmb': {'date_time': [2026, 10, 27, 10, 45, 58], 'mode': 33206}, 'dfthcsos/angle_k_11.kmb': {'date_time': [2026, 10, 27, 10, 45, 58], 'mode': 33206}, 'dfthcsos/angle_k_12.kmb': {'date_time': [2026, 10, 27, 10, 45, 58], 'mode': 33206}, 'dfthcsos/angle_k_13.kmb': {'date_time': [2026, 10, 27, 10, 45, 58], 'mode': 33206}, 'dfthcsos/angle_k_14.kmb': {'date_time': [2026, 10, 27, 10, 45, 58], 'mode': 33206}, 'dfthcsos/angle_k_2.kmb': {'date_time': [2026, 10, 27, 10, 45, 58], 'mode': 33206}, 'dfthcsos/angle_k_3.kmb': {'date_time': [2026, 10, 27, 10, 45, 58], 'mode': 33206}, 'dfthcsos/angle_k_4.kmb': {'date_time': [2026, 10, 27, 10, 45, 58], 'mode': 33206}, 'dfthcsos/angle_k_5.kmb': {'date_time': [2026, 10, 27, 10, 45, 58], 'mode': 33206}, 'dfthcsos/angle_k_6.kmb': {'date_time': [2026, 10, 27, 10, 45, 58], 'mode': 33206}, 'dfthcsos/angle_k_7.kmb': {'date_time': [2026, 10, 27, 10, 45, 58], 'mode': 33206}, 'dfthcsos/angle_k_8.kmb': {'date_time': [2026, 10, 27, 10, 45, 58], 'mode': 33206}, 'dfthcsos/angle_k_9.kmb': {'date_time': [2026, 10, 27, 10, 45, 58], 'mode': 33206}, 'dfthcsos/submission.npz': {'date_time': [2026, 10, 27, 16, 43, 30], 'mode': 33188}, 'dqi.kmb': {'date_time': [2026, 10, 27, 16, 43, 54], 'mode': 33188}, 'invsqrt.kmb': {'date_time': [2026, 10, 27, 10, 45, 58], 'mode': 33206}}  # each member's stamp and mode in the bundle's zip
ASSEMBLED = {'sha256': 'b28c1e4ff9e49b6a57f08c3ff0b0ba6ba42d5ee56a0282d3063d0eb643c6ce43', 'bytes': 30462493}  # the bundle zip: the exact bytes this notebook must write
OWN_LANE = REC['own_lane_on']  # verify our own members separately when the zip holds none of ours for a job
FIELD_LIVE = False  # chapter 8 reads the recorded field runs; True would re-run public files (more minutes)
TASKS = ('invsqrt', 'dqi', 'dfthcsos')
JOB = {'invsqrt': 'fusion', 'dqi': 'message repair', 'dfthcsos': 'chemistry'}
INPUT = pathlib.Path(os.environ.get('QCC33_INPUT', '/kaggle/input'))
WORKING = pathlib.Path(os.environ.get('QCC33_WORKING', '/kaggle/working'))
SCRATCH = pathlib.Path(os.environ.get('QCC33_SCRATCH', '/tmp/qcc33'))
FIGDIR = WORKING / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)
apply()
plt.rcParams.update({'figure.dpi': 110, 'axes.titlesize': 12.5})


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def show(fig, name, source):
    fig.text(0.0, -0.32 / fig.get_figheight(), 'source: ' + source, fontsize=8, color=MUTED, ha='left', va='top')
    fig.savefig(FIGDIR / name, bbox_inches='tight', facecolor='white')
    plt.show()


print(f'{platform.python_version()} on {platform.machine()}, {os.cpu_count()} CPUs, start {START_UTC}')

# CELL 2
# R1 locate and pin: each member is found under the inputs by name and accepted only by its SHA-256.
def materialise(arc, pin, root):
    dst = root / arc
    dst.parent.mkdir(parents=True, exist_ok=True)
    tried = []
    for p in sorted(INPUT.rglob(pin['file'])):
        if not p.is_file():
            continue
        if pin.get('member'):  # a member of a public kernel's submission.zip
            with zipfile.ZipFile(p) as z:
                if pin['member'] not in z.namelist():
                    tried.append((str(p), 'no such member'))
                    continue
                with z.open(pin['member']) as s, open(dst, 'wb') as d:
                    shutil.copyfileobj(s, d, 1 << 22)
        elif pin.get('codec') == 'xz':
            with lzma.open(p) as s, open(dst, 'wb') as d:
                shutil.copyfileobj(s, d, 1 << 22)
        else:
            shutil.copyfile(p, dst)
        got = sha256(dst)
        if got == pin['sha256']:
            return p, dst
        tried.append((str(p), got))
    dst.unlink(missing_ok=True)
    raise RuntimeError(f"STOP: {arc}: no input file {pin['file']} with SHA-256 {pin['sha256']} "
                       f"(source {pin['source']}); tried {tried}")


SUB = SCRATCH / 'submission'
shutil.rmtree(SUB, ignore_errors=True)
SUB.mkdir(parents=True)
t = time.monotonic()
FOUND = {}
for arc, pin in sorted(PINS.items()):
    src, dst = materialise(arc, pin, SUB)
    FOUND[arc] = str(src)
    print(f"{arc:<26} {dst.stat().st_size:>12,} B  sha256 {pin['sha256'][:12]}...  ok  <- {src}")
extra = sorted(str(p.relative_to(SUB)) for p in SUB.rglob('*') if p.is_file() and str(p.relative_to(SUB)) not in PINS)
assert not extra, extra
PIN_SECONDS = time.monotonic() - t
print(f'{len(PINS)} of {len(PINS)} members found and matched their pins in {PIN_SECONDS:.1f} s')

# CELL 3
# R2 write: the competition layout, sorted names, the bundle's own stamps and modes, deflate level 6.
OUT_ZIP = WORKING / 'submission.zip'
t = time.monotonic()
with zipfile.ZipFile(OUT_ZIP, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for name in sorted(PINS):
        info = zipfile.ZipInfo(name, date_time=tuple(ZIPMETA[name]['date_time']))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = ZIPMETA[name]['mode'] << 16
        info.file_size = (SUB / name).stat().st_size
        with open(SUB / name, 'rb') as s, z.open(info, 'w') as d:
            shutil.copyfileobj(s, d, 1 << 22)
with zipfile.ZipFile(OUT_ZIP) as z:
    assert z.testzip() is None and sorted(z.namelist()) == sorted(PINS)
ZIP_SHA = sha256(OUT_ZIP)
SAME = ZIP_SHA == ASSEMBLED['sha256']
ZIP_SECONDS = time.monotonic() - t
print(f"{OUT_ZIP}: {OUT_ZIP.stat().st_size:,} B, sha256 {ZIP_SHA}; same bytes as {REC['bundle']}: {SAME} "
      f'({ZIP_SECONDS:.1f} s)')

# CELL 4
# R3 verify: the hosts' metric code, unchanged, in a child process, on the Public split of our own local run of that
# code on these bytes (same salts, same reference costs). R4 compare: each job's cost against that record.
comp = sorted({p.parents[2] for p in INPUT.rglob('verify_submission_invsqrt.py')})
comp = [c for c in comp if (c / 'kickmix').is_dir() and 'competitions' in c.parts] or comp  # the competition data
assert len(comp) == 1, comp
COMP = comp[0]
KICKMIX_DIR = os.environ.get('QCC33_KICKMIX', str(COMP / 'kickmix'))
VERIFIERS_DIR = str(COMP / 'verifiers')
EXP = REC['expected']
for t_ in TASKS:  # our local runner made each salt from a public formula; print whether these are those
    formula = hashlib.sha256(f'33-metric-local-submission.zip-Public-{t_}'.encode()).hexdigest()[:40]
    print(f"salt {t_:<9} {EXP['salts'][t_]}  (from the runner's formula: {formula == EXP['salts'][t_]})")
METRIC_SRC = '"""Kaggle metric for 161954: verifiable quantum circuit compilation benchmarks.\n\nA submission is an archive holding one entry per benchmark, with every circuit in either the text\n(.kmx) or the binary (.kmb) kickmix format:\n\n    invsqrt.kmx or invsqrt.kmb   Benchmark I: phasing by beta / sqrt(x), at n_p = 10.\n    dqi.kmx or dqi.kmb           Benchmark II: the DQI Reed-Solomon decoder, RS(4095, 4025) over GF(2^12).\n    dfthcsos/                    Benchmark III: the DFTHC-SOS block encoding of FeMoCo76o113e, as the\n                                 ANGLE_k circuits angle_k_0 ... angle_k_<n-1>, each .kmx or .kmb,\n                                 and submission.npz, holding the factorization and the Givens tree.\n\nEach benchmark is checked by its verifier, from the competition data, which simulates the circuits on many test inputs,\nrejects the submission if any test fails, and otherwise returns its cost (lower is better). Each\nleaderboard split seeds the simulations with its own secret salt, so every benchmark is verified once\nper split. A split\'s score is the geometric mean over the benchmarks of `baseline_cost / cost`: the\nhosts\' reference submissions score 1.0, and halving every benchmark\'s cost scores 2.0.\n\nThe verifiers and kickmix come from the competition data itself, attached to the metric, so it scores\nwith exactly the files participants download: verifiers/ and kickmix/, built by the 161954 prep\npipeline from pinned commits of the hosts\' code and Qualtran. kickmix.so is compiled for that\nbuild\'s Kaggle image, so the metric must run in that same image. To run the metric locally instead,\ninstall kickmix and the verifiers; local installs take precedence over the competition data.\n"""\n\nimport json\nimport math\nimport pathlib\nimport random\nimport stat\nimport sys\nimport time\nimport tokenize\nimport zipfile\nimport zlib\nfrom collections.abc import Callable, Mapping\nfrom typing import Any\n\nimport numpy as np\nimport pandas as pd\n\n# kickmix/ and verifiers/ in the competition data, at the 161954 prep pipeline\'s COMPETITION_DATA_MOUNT.\nKICKMIX_DIR = \'/kaggle/input/competitions/google-verifiable-quantum-circuit-compilation/kickmix\'\nVERIFIERS_DIR = \'/kaggle/input/competitions/google-verifiable-quantum-circuit-compilation/verifiers\'\n# Where the orchestrator, and run_metric_locally, put the solution and metadata files.\nEVAL_INPUTS_DIR = pathlib.Path(\'/kaggle/input/competition_evaluation\')\n\n# Every kickmix instruction, as named by Circuit.operation_statistics. Listing them rejects any\n# instruction a future Qualtran adds until it is reviewed. INFO_TAG is a no-op in simulation;\n# kickmix\'s CircuitBuilder emits it around every iadd and isub. DEBUG_PRINT is a no-op too, as every\n# verifier\'s simulator ignores it, but its qubit and bit ids still count toward the circuit\'s.\nALLOWED_INSTRUCTIONS = {\n    \'X\', \'CX\', \'CCX\', \'Z\', \'CZ\', \'CCZ\', \'Z_POW\', \'NEG\', \'SWAP\', \'CSWAP\', \'R\', \'HMR\', \'BIT_INVERT\', \'BIT_STORE0\', \'BIT_STORE1\',\n    \'PUSH_CONDITION\', \'POP_CONDITION\', \'INFO_TAG\', \'DEBUG_PRINT\',\n}  # fmt: skip\n# Qubit and bit ids must be below this. Verifier memory grew by ~1.5 KB per bit (dqi) and ~1.1 KB\n# per qubit (invsqrt) with the verifiers at c636a96, so this capped the growth at ~1 GB; at 03027f8,\n# dqi\'s grows by only ~0.2 KB per qubit or bit, linearly rather than with their product. The\n# baselines use ids below 50,000. Raise it if a harder benchmark needs more.\nMAX_QUBIT_OR_BIT_ID = 2**30 // 1536\n# The arrays read from a dfthcsos submission.npz may total at most this many bytes, ~30 times the\n# ~2 MB the reference factorization and tree take.\nMAX_NPZ_BYTES = 2**26\n\n\nclass ParticipantVisibleError(Exception):\n    pass\n\n\ndef read_circuit(path: pathlib.Path, max_registers: int) -> Any:\n    """Return a submitted .kmx or .kmb circuit as a kickmix.Circuit; raise ValueError unless it is fit to verify.\n\n    kickmix validates a file as it parses it, within limits that keep parsing cheap, but simulating\n    takes memory that grows with the largest qubit and bit ids, which kickmix allows up to ~5e8,\n    so those are capped at MAX_QUBIT_OR_BIT_ID. kickmix also allows a million registers, at\n    ~110 bytes each even if empty, so a circuit may have at most `max_registers`, the most its\n    benchmark needs. Every register must be named: the dfthcsos verifier crashes on an unnamed one.\n    """\n    import kickmix  # ty: ignore[unresolved-import]  # In kickmix/, which generate_standard_submission put on sys.path.\n\n    circuit = kickmix.Circuit.from_file(path, format=path.suffix.removeprefix(\'.\'))\n    unsupported = set(circuit.operation_statistics()) - ALLOWED_INSTRUCTIONS\n    if unsupported:\n        raise ValueError(f\'unsupported instructions {sorted(unsupported)}\')\n    largest_id = max(circuit.num_qubits, circuit.num_bits) - 1\n    if largest_id >= MAX_QUBIT_OR_BIT_ID:\n        raise ValueError(f\'qubit or bit id {largest_id} is above the limit of {MAX_QUBIT_OR_BIT_ID - 1}\')\n    if circuit.num_registers > max_registers:\n        raise ValueError(f\'{circuit.num_registers} registers, above the limit of {max_registers} for this benchmark\')\n    if not all(isinstance(name, str) for name in circuit.register_data):\n        raise ValueError(\'every register needs a name, given by its REGISTER line\')\n    return circuit\n\n\ndef read_npz(path: pathlib.Path, spec: Mapping[str, tuple[str, tuple[int | str, ...]]]) -> dict[str, np.ndarray]:\n    """Return the arrays `spec` names from an .npz; raise ValueError unless each is a plain number array as specified.\n\n    `spec` maps each key to the numpy dtype kinds it may have, such as \'f\' for floats, and its shape,\n    in which a string names a dimension of any positive size that must agree wherever it recurs.\n\n    Each array\'s header is a Python literal of its dtype and shape, which np.load trusts: a few KB\n    of deflated zeros can claim a terabyte-sized array, and the dtype can be anything numpy can\n    describe, from structured and nested types to Python objects that need unpickling. So the\n    headers are read first, and the arrays loaded only if each is a plain number type of its kind,\n    not big-endian, with its shape, and they total at most MAX_NPZ_BYTES. Only format 1.0 headers,\n    which numpy.savez writes, are accepted, and numpy refuses any over 10 KB before parsing it.\n    """\n    dimensions: dict[str, int] = {}\n    total = 0\n    with zipfile.ZipFile(path) as archive:\n        members = {name.removesuffix(\'.npy\'): name for name in archive.namelist()}\n        missing = [key for key in spec if key not in members]\n        if missing:\n            raise ValueError(f\'{path.name} has no {missing}\')\n        for key, (kinds, expected_shape) in spec.items():\n            with archive.open(members[key]) as member:\n                if np.lib.format.read_magic(member) != (1, 0):\n                    raise ValueError(f\'{key} in {path.name} is not in .npy format 1.0, which numpy.savez writes\')\n                shape, _, dtype = np.lib.format.read_array_header_1_0(member)\n            if dtype.kind not in kinds or dtype.names is not None or dtype.subdtype is not None or dtype.byteorder == \'>\':\n                raise ValueError(f\'{key} in {path.name} has dtype {dtype}, not a plain number type of numpy kind {kinds!r}\')\n            matches = len(shape) == len(expected_shape) and all(\n                size > 0 and size == (dimensions.setdefault(expected, size) if isinstance(expected, str) else expected)\n                for size, expected in zip(shape, expected_shape, strict=True)\n            )\n            if not matches:\n                raise ValueError(f\'{key} in {path.name} has shape {shape}, not {expected_shape} with its named dimensions as in {dimensions}\')\n            total += math.prod(shape) * dtype.itemsize\n    if total > MAX_NPZ_BYTES:\n        raise ValueError(f\'the arrays in {path.name} take {total:,} bytes, above the limit of {MAX_NPZ_BYTES:,}\')\n    with np.load(path, allow_pickle=False) as npz:\n        return {key: np.array(npz[key]) for key in spec}\n\n\ndef find_one(root: pathlib.Path, name: str, description: str, is_dir: bool = False) -> pathlib.Path:\n    """The one path called `name` (a glob) under `root`, where the submission unpacked; raise ParticipantVisibleError otherwise."""\n    # macOS Finder\'s Compress adds __MACOSX/, which mirrors every directory, with an AppleDouble ._ file per file.\n    paths = [path for path in root.rglob(name) if path.is_dir() == is_dir and \'__MACOSX\' not in path.relative_to(root).parts]\n    if len(paths) != 1:\n        raise ParticipantVisibleError(f\'The submission must contain exactly one {description}, found {len(paths)}.\')\n    return paths[0]\n\n\ndef generate_standard_submission(submission_dir: str) -> None:\n    """Verify the submission under every split\'s salt, and write each verification\'s cost as submission.csv.\n\n    `submission_dir` is where the orchestrator unpacked the submission archive. Every split\'s verifications run here, in random order, before the orchestrator calls score() for\n    either split, so neither a failure\'s timing nor its message shows which split it failed on.\n    Every file is read and checked before any verifier runs, so a malformed file fails in seconds\n    rather than after another benchmark\'s verification. The verifiers run one at a time, as the\n    invsqrt and dqi verifiers spread their own simulation across processes.\n\n    The orchestrator passes neither the solution nor the metadata in, so the solution is read from\n    where it puts it; this metric\'s own solution names its row id column `row_id`. Each row\'s\n    `seed_salt` is merged by its verifier with the submission\'s hash to seed the simulation, so a\n    submission always faces the same tests, but participants can\'t compute them. Unsalted, the seed\n    would let a team whose circuit fails on a few inputs re-roll its tests offline with cost-free\n    edits until they miss those inputs; with the salt, each re-roll costs a submission. A salt that\n    is missing or short is refused, as it would leave the seed unsalted or guessable.\n\n    kickmix and the verifiers are imported here rather than at the top because the orchestrator imports\n    this module during validation, when the competition data isn\'t mounted.\n    """\n    solution_paths = list(EVAL_INPUTS_DIR.glob(\'solution.*\'))\n    if len(solution_paths) != 1:\n        raise ValueError(f\'Expected one solution.csv or solution.parquet in {EVAL_INPUTS_DIR}, found {[path.name for path in solution_paths]}\')\n    [solution_path] = solution_paths\n    solution = pd.read_csv(solution_path) if solution_path.suffix == \'.csv\' else pd.read_parquet(solution_path)\n    scored = solution[solution[\'Usage\'] != \'Ignored\']\n    # score()\'s hyperparameters, which the orchestrator passes only to score().\n    hyperparameters = json.loads((EVAL_INPUTS_DIR / \'metadata.json\').read_text()).get(\'Hyperparameters\') or {}\n    shot_divisor = int(hyperparameters.get(\'verification_shot_divisor\', 1))\n    for salt in scored[\'seed_salt\']:\n        if not isinstance(salt, str) or len(salt) < 32:\n            raise ValueError(f\'Every scored solution row needs a seed_salt of at least 32 characters, got {salt!r}\')\n\n    # Appended, not prepended, so local installs of kickmix and the verifiers win when running locally.\n    sys.path += [KICKMIX_DIR, VERIFIERS_DIR]\n    from dfthcsos import verify as dfthcsos_verify  # ty: ignore[unresolved-import]  # These live in verifiers/.\n    from dfthcsos.scoreinfra.reference import FACTORIZATION_KEYS, ORBITALS  # ty: ignore[unresolved-import]\n    from dqi import verify_submission_dqi  # ty: ignore[unresolved-import]\n    from invsqrt import verify_submission_invsqrt  # ty: ignore[unresolved-import]\n\n    root = pathlib.Path(submission_dir)\n\n    def read(path: pathlib.Path, reader: Callable[[pathlib.Path, Any], Any], spec: Any) -> Any:\n        # tar and 7z submissions can unpack links, which would have this read the host\'s files, and\n        # FIFOs and devices, which would block or never end.\n        status = path.lstat()\n        if not stat.S_ISREG(status.st_mode) or status.st_nlink > 1 or not path.resolve().is_relative_to(root.resolve()):\n            raise ParticipantVisibleError(f\'{path.relative_to(root)} must be a regular file, not a link or a special file.\')\n        try:\n            return reader(path, spec)\n        # What kickmix, zipfile, zlib and numpy raise on malformed files, found by fuzzing an .npz.\n        except (ValueError, RuntimeError, OSError, EOFError, zipfile.BadZipFile, zlib.error, tokenize.TokenError) as err:\n            raise ParticipantVisibleError(f\'{path.relative_to(root)} is invalid: {err}\') from err\n\n    batches = max(1, verify_submission_dqi.DEFAULT_NUM_SHOTS // verify_submission_dqi.DEFAULT_BATCH_SIZE // shot_divisor)\n    dqi_spec = verify_submission_dqi.ProblemSpecDQI.benchmark_instance(num_shots=batches * verify_submission_dqi.DEFAULT_BATCH_SIZE)\n    # Each circuit\'s most registers: invsqrt\'s x, and dqi\'s S_i and E_j.\n    invsqrt = read(find_one(root, \'invsqrt.km[xb]\', \'invsqrt.kmx or invsqrt.kmb\'), read_circuit, 1)\n    dqi = read(find_one(root, \'dqi.km[xb]\', \'dqi.kmx or dqi.kmb\'), read_circuit, dqi_spec.n + dqi_spec.m)\n    dfthcsos_dir = find_one(root, \'dfthcsos\', \'dfthcsos directory\', is_dir=True)\n    angle_files = list(dfthcsos_dir.glob(\'angle_k_*.km[xb]\'))\n    angle_paths = {path.stem: path for path in angle_files}\n    # A circuit given in both formats leaves fewer stems than files, so the two can\'t match.\n    stems = [f\'angle_k_{k}\' for k in range(len(angle_files))]\n    if not stems or sorted(angle_paths) != sorted(stems):\n        raise ParticipantVisibleError(\'dfthcsos/ must hold the circuits angle_k_0 ... angle_k_<n-1>, each once, as .kmx or .kmb.\')\n    # An ANGLE_k circuit\'s `selection` and `word_j` registers each hold a qubit, of at most QUBIT_CAP.\n    # The verifier allows other registers too, but none are needed.\n    angle_circuits = [read(angle_paths[stem], read_circuit, dfthcsos_verify.QUBIT_CAP) for stem in stems]\n    # The factorization (see dfthcsos.scoreinfra.reference), with R factors of B vectors over the N\n    # orbitals and C copies, and the Givens tree\'s N - 1 edges.\n    npz_spec = {\n        \'U\': (\'f\', (\'R\', \'B\', ORBITALS)),\n        \'W\': (\'f\', (\'R\', \'B\', \'C\')),\n        \'H2_I\': (\'f\', (\'R\', \'C\')),\n        \'H1_I\': (\'f\', ()),\n        \'Hsym_lower\': (\'f\', (ORBITALS * (ORBITALS + 1) // 2,)),\n        \'tree_edges\': (\'iu\', (ORBITALS - 1, 2)),\n    }\n    if set(npz_spec) != {*FACTORIZATION_KEYS, \'tree_edges\'}:\n        raise ValueError(f\'The dfthcsos verifier reads the factorization arrays {FACTORIZATION_KEYS}, which npz_spec must cover exactly\')\n    npz = read(find_one(dfthcsos_dir, \'submission.npz\', \'dfthcsos/submission.npz\'), read_npz, npz_spec)\n    tree_edges = npz.pop(\'tree_edges\')\n\n    # Benchmark -> its verifier run, given the salt.\n    verifications: dict[str, Callable[[str], dict[str, Any]]] = {\n        \'invsqrt\': lambda salt: verify_submission_invsqrt.verify(invsqrt, verify_submission_invsqrt.ProblemSpecInvSqrt.for_n_p(10), salt=salt),\n        \'dqi\': lambda salt: verify_submission_dqi.verify(dqi, dqi_spec, salt=salt),\n        \'dfthcsos\': lambda salt: dfthcsos_verify.verify(\n            angle_circuits, npz, tree_edges, salt=salt, shots_per_address=max(1, dfthcsos_verify.SHOTS_PER_ADDRESS // shot_divisor)\n        ),\n    }\n    runs = list(zip(scored[\'row_id\'], scored[\'benchmark\'], scored[\'seed_salt\'], strict=True))\n    random.shuffle(runs)\n    # Ignored rows aren\'t scored, but the orchestrator still expects a submission row for each.\n    costs, seconds = dict.fromkeys(solution[\'row_id\'], 0.0), dict.fromkeys(solution[\'row_id\'], 0.0)\n    for row_id, benchmark, salt in runs:\n        start = time.monotonic()\n        try:\n            costs[row_id] = float(verifications[benchmark](salt)[\'score\'])\n        # The exceptions the verifiers\' command lines report as a failed submission; dfthcsos\'s\n        # InvalidCircuits is a ValueError.\n        except (AssertionError, ValueError, RuntimeError, KeyError) as err:\n            raise ParticipantVisibleError(f\'The {benchmark} submission failed verification: {err}\') from err\n        if not costs[row_id] > 0:\n            raise ParticipantVisibleError(f\'The {benchmark} submission has non-positive cost {costs[row_id]}.\')\n        seconds[row_id] = time.monotonic() - start\n    pd.DataFrame({\'row_id\': list(costs), \'cost\': list(costs.values()), \'seconds\': list(seconds.values())}).to_csv(\'submission.csv\', index=False)\n\n\ndef score(\n    solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, verification_shot_divisor: int = 1\n) -> tuple[float, dict[str, Any]]:\n    """Return the geometric mean over the benchmarks of `baseline_cost / cost`, where higher is better.\n\n    `verification_shot_divisor` is for testing only, and must stay 1 on Kaggle. generate_standard_submission\n    reads it, and divides the dqi and dfthcsos verifiers\' default numbers of shots by it, which makes\n    their verification that much faster and weaker, and changes their costs, so that the reference\n    submissions no longer score 1.0. invsqrt\'s verifier is the fastest, and needs a multiple of 64\n    shots per input to fill its batches, its default, so it keeps every shot.\n\n    The solution has a row per benchmark and leaderboard split, holding the split\'s secret\n    `seed_salt` and the `baseline_cost` its benchmark\'s reference submission scored under that salt.\n    `submission` is generate_standard_submission\'s output, which holds each row\'s `cost`, as the\n    verification runs there. Returns each benchmark\'s cost, improvement and verification time as\n    details.\n    """\n    merged = solution.merge(submission, on=row_id_column_name, validate=\'one_to_one\')\n    details = {\n        benchmark: {\'cost\': cost, \'improvement_over_baseline\': baseline_cost / cost, \'seconds\': run_seconds}\n        for benchmark, cost, baseline_cost, run_seconds in zip(\n            merged[\'benchmark\'], merged[\'cost\'], merged[\'baseline_cost\'], merged[\'seconds\'], strict=True\n        )\n    }\n    ratios = [benchmark_details[\'improvement_over_baseline\'] for benchmark_details in details.values()]\n    return math.prod(ratios) ** (1 / len(ratios)), details\n'  # the hosts' metric notebook, code cell, unchanged
RUNNER_SRC = '\n"""33 | Runs the hosts\' metric code (google-quantum-ai-metric, unchanged) on the unpacked submission in a child\nprocess, and saves each verifier\'s full report as a receipt. Config: one JSON file, argv[1]."""\nimport functools, hashlib, json, os, pathlib, resource, sys, time, types\n\n\ndef main(cfg_path):\n    cfg = json.loads(pathlib.Path(cfg_path).read_text())\n    src = pathlib.Path(cfg[\'metric_py\']).read_text()\n    assert hashlib.sha256(src.encode()).hexdigest() == cfg[\'metric_sha256\'], \'the metric source is not the pinned one\'\n    sys.path += [cfg[\'kickmix_dir\'], cfg[\'verifiers_dir\']]  # appended, as the metric does\n    from invsqrt import verify_submission_invsqrt as v_inv\n    from dqi import verify_submission_dqi as v_dqi\n    from dfthcsos import verify as v_chem\n    receipts = pathlib.Path(cfg[\'receipts\'])\n    receipts.mkdir(parents=True, exist_ok=True)\n\n    def keep(task, module):\n        original = module.verify\n\n        @functools.wraps(original)\n        def run(*args, **kwargs):\n            t = time.monotonic()\n            report = original(*args, **kwargs)\n            seconds = time.monotonic() - t\n            out = json.loads(json.dumps(report, default=lambda o: o.item() if hasattr(o, \'item\') else str(o)))\n            out[\'_salt\'], out[\'_seconds\'] = kwargs.get(\'salt\'), seconds\n            (receipts / f\'{task}.json\').write_text(json.dumps(out, indent=1))\n            print(f"{task:<9} {out.get(\'status\')}  cost {out.get(\'score\')}  {seconds:.1f} s", flush=True)\n            return report\n        module.verify = run\n\n    keep(\'invsqrt\', v_inv)\n    keep(\'dqi\', v_dqi)\n    keep(\'dfthcsos\', v_chem)\n    metric = types.ModuleType(\'google_quantum_ai_metric\')\n    exec(compile(src, \'google-quantum-ai-metric\', \'exec\'), metric.__dict__)\n    metric.KICKMIX_DIR, metric.VERIFIERS_DIR = cfg[\'kickmix_dir\'], cfg[\'verifiers_dir\']\n    metric.EVAL_INPUTS_DIR = pathlib.Path(cfg[\'eval_dir\'])\n    os.chdir(cfg[\'eval_dir\'])  # generate_standard_submission writes submission.csv here\n    t0 = time.monotonic()\n    metric.generate_standard_submission(cfg[\'unpacked\'])\n    import pandas as pd\n    submission = pd.read_csv(\'submission.csv\')\n    solution = pd.read_csv(pathlib.Path(cfg[\'eval_dir\']) / \'solution.csv\')\n    score, details = metric.score(solution, submission, \'row_id\')\n    out = {\'score\': score, \'details\': details, \'wall_seconds\': time.monotonic() - t0,\n           \'maxrss_self_kib\': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,\n           \'maxrss_children_kib\': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}\n    pathlib.Path(cfg[\'out\']).write_text(json.dumps(out, indent=1, default=float))\n    print(json.dumps({\'score\': score, \'wall_seconds\': out[\'wall_seconds\']}), flush=True)\n\n\nif __name__ == \'__main__\':\n    main(sys.argv[1])\n'
METRIC_SHA = hashlib.sha256(METRIC_SRC.encode()).hexdigest()


def run_metric(unpacked, tasks, tag):
    eval_dir, receipts, out = SCRATCH / f'eval_{tag}', WORKING / f'receipts{tag}', SCRATCH / f'metric_{tag}.json'
    eval_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{'row_id': f'Public_{t}', 'benchmark': t, 'seed_salt': EXP['salts'][t],
                   'baseline_cost': EXP['baseline'][t], 'Usage': 'Public'} for t in tasks]
                 ).to_csv(eval_dir / 'solution.csv', index=False)
    (eval_dir / 'metadata.json').write_text(json.dumps({'Hyperparameters': {}}))
    (SCRATCH / 'google_quantum_ai_metric.py').write_text(METRIC_SRC)
    (SCRATCH / 'runner.py').write_text(RUNNER_SRC)
    cfg = {'metric_py': str(SCRATCH / 'google_quantum_ai_metric.py'), 'metric_sha256': METRIC_SHA,
           'kickmix_dir': KICKMIX_DIR, 'verifiers_dir': VERIFIERS_DIR, 'eval_dir': str(eval_dir),
           'unpacked': str(unpacked), 'receipts': str(receipts), 'out': str(out)}
    (SCRATCH / f'cfg_{tag}.json').write_text(json.dumps(cfg))
    t = time.monotonic()
    proc = subprocess.Popen([sys.executable, '-u', str(SCRATCH / 'runner.py'), str(SCRATCH / f'cfg_{tag}.json')],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end='')
    rc = proc.wait()
    assert rc == 0, f'the metric run exited with {rc}'
    res = json.loads(out.read_text())
    res['seconds_outside'] = time.monotonic() - t
    reports = {t: json.loads((receipts / f'{t}.json').read_text()) for t in tasks}
    return res, reports


RESULT, REPORTS = run_metric(SUB, TASKS, '')
RSS_GIB = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / (2**30 if sys.platform == 'darwin' else 2**20)  # KiB on Linux, bytes on macOS
SCORE = RESULT['score']
REL = {t: abs(REPORTS[t]['score'] - EXP['costs'][t]) / EXP['costs'][t] for t in TASKS}
REL_SCORE = abs(SCORE - EXP['score']) / EXP['score']
for t in TASKS:
    assert REPORTS[t]['status'] == 'PASS' and REPORTS[t]['_salt'] == EXP['salts'][t], t
    print(f"{t:<9} PASS  cost {REPORTS[t]['score']:,.6f}  record {EXP['costs'][t]:,.6f}  rel diff {REL[t]:.1e}")
print(f"score by the hosts' metric code: {SCORE:.5f} (record {EXP['score']:.5f}, rel diff {REL_SCORE:.1e}); "
      f"metric {RESULT['wall_seconds']:.0f} s; peak RSS of the largest child {RSS_GIB:.2f} GiB")

# CELL 5
# R6 own lane: when the zip holds no member of ours for a job, our best own member for it is verified the same way.
OWN_REPORTS = {}
missing = REC['own_missing']
if OWN_LANE and missing:
    OWN = SCRATCH / 'own'
    shutil.rmtree(OWN, ignore_errors=True)
    shutil.copytree(SUB, OWN)
    for arc, pin in OWN_PINS.items():
        if arc.split('/')[0].split('.')[0] in missing:
            (OWN / arc).unlink(missing_ok=True)
            materialise(arc, pin, OWN)
    _, OWN_REPORTS = run_metric(OWN, tuple(missing), '_ours')
    print('own lane:', {t: (r['status'], r['score']) for t, r in OWN_REPORTS.items()})
else:
    print('own lane: off; our earlier members are quoted from their records (chapters 5 to 7), not re-run here')

# CELL 6
R = {t: REPORTS[t] for t in TASKS}
ref = REC['ref']
ri, rd, rc = R['invsqrt'], R['dqi'], R['dfthcsos']
k_run = rd['value_errors'] + rd['phase_errors']
ratio = {t: EXP['baseline'][t] / R[t]['score'] for t in TASKS}


def money(x):
    return f'{x:,.0f}' if x < 1e7 else f'{x:.3e}'


def stamp(s):
    return f'<span style="color:{ACCENT};font-weight:700">{s}</span>'


rows = [
    ('JOB 1  FUSION, phase by 1/sqrt(x)', ri['verified_shots'], '',
     f"{ref['invsqrt']['q']} seats x {ref['invsqrt']['c']:,.2f} stamps", ref['invsqrt']['cost'],
     f"{ri['num_qubits']} seats x {ri['toffoli_equivalent']:,.2f} stamps", ri['score'], 'invsqrt'),
    ('JOB 2  MESSAGE REPAIR, Reed-Solomon decoder', rd['verified_shots'], f', {k_run} failed',
     f"R {ref['dqi']['r']:.5f} x {ref['dqi']['q']:,}^2 seats x {ref['dqi']['t']:,.0f} stamps", ref['dqi']['cost'],
     f"R {rd['repetition_factor']:.5f} x {rd['num_qubits']}^2 seats x {rd['toffoli_count']:,.0f} stamps", rd['score'], 'dqi'),
    ('JOB 3  CHEMISTRY, FeMoCo lookups and rotations', rc['verified_shots'], '',
     f"volume {ref['dfthcsos']['lam']:.2f} x {ref['dfthcsos']['t']:,.0f} stamps x {ref['dfthcsos']['q']} seats",
     ref['dfthcsos']['cost'], f"volume {rc['lambda_eff']:.2f} x {rc['toffoli_count']:,.0f} stamps x {rc['num_qubits']} seats",
     rc['score'], 'dfthcsos'),
]
W = 92
lines = [f"RECEIPT  Google Quantum AI, Verifiable Quantum Circuit Compilation{START_UTC:>{W - 64}}", '=' * W]
for head, shots, fails, g_bill, g_cost, o_bill, o_cost, t in rows:
    lines.append(f"{head:<58}" + stamp('PASS') + f"  {shots:,} checks{fails}")
    lines.append(f"  Google's bill   {g_bill:<58}= {money(g_cost):>12}")
    lines.append(stamp(f"  our bill        {o_bill:<58}= {money(o_cost):>12}"))
    lines.append(f"{'':<64}" + stamp(f"{ratio[t]:>8.2f}x cheaper"))
    lines.append('')
lines += ['-' * W,
          f"TOTAL  geometric mean of the three ratios, by the hosts' metric code, this run {stamp(f'{SCORE:>10.5f}')}",
          f"same bytes as our submission {REC['bundle']} ({ASSEMBLED['sha256'][:16]}...): {stamp(str(SAME)) if SAME else 'False'}"
          + (f", LB {REC['lb']}" if REC['lb'] else ''),
          f"cost difference to the record of the same code, bytes and salts: at most {max(REL.values()):.1e}",
          f"run so far {time.monotonic() - T0:,.0f} s on {os.cpu_count()} CPUs; peak RSS {RSS_GIB:.2f} GiB"]
display(HTML('<pre style="font-family:Menlo,Consolas,monospace;font-size:12.5px;line-height:1.35;background:#fff;'
             'border:2px dashed #8b949e;padding:14px 18px;color:#1f2328;white-space:pre;overflow-x:auto">'
             + '\n'.join(lines) + '</pre>'))

fig, ax = plt.subplots(figsize=(9.6, 2.6))
y = np.arange(3)[::-1]
for yi, t in zip(y, TASKS):
    ax.barh(yi + 0.17, 1.0, height=0.32, color='#adb5bd')
    ax.barh(yi - 0.17, 1 / ratio[t], height=0.32, color=ACCENT)
    ax.text(1 / ratio[t] * 1.08, yi - 0.17, f'ours: {ratio[t]:.2f}x cheaper', va='center', fontsize=10, color=ACCENT)
    ax.text(1.0 * 1.08, yi + 0.17, "Google's bill", va='center', fontsize=9.5, color=MUTED)
ax.set_xscale('log')
ax.set_xlim(0.02, 4)
ax.set_yticks(y, [JOB[t] for t in TASKS])
ax.grid(axis='y', visible=False)
ax.set_xlabel("bill, as a share of Google's reference bill (log scale)")
show(fig, 'f0_receipt_bars.png', '/kaggle/working/receipts/*.json (this run); sample_invsqrt.verify.json, sample_dqi.json, '
     "sample_dfthcsos.json (reference bills); data/qcc/runs/" + REC["bundle"] + "/metric_local/metric_local.json")

# CELL 9
def run_toy(circuit, a, b, coin):
    """Run a 4-wire circuit on inputs a, b. coin() gives the random bit an HMR measurement returns."""
    w, sign, m = [a, b, 0, 0], 1, 0
    for gate, *q in circuit:
        if gate == 'X':
            w[q[0]] ^= 1
        elif gate == 'CX':
            w[q[1]] ^= w[q[0]]
        elif gate == 'CCX':  # the one gate on the bill: a Toffoli
            w[q[2]] ^= w[q[0]] & w[q[1]]
        elif gate == 'HMR':  # measure in the X basis and reset: random m, phase (-1)^(m * wire)
            m = coin()
            if m and w[q[0]]:
                sign = -sign
            w[q[0]] = 0
        elif gate == 'CZ_IF_M':  # classically controlled CZ: the cheap phase fix after HMR
            if m and w[q[0]] and w[q[1]]:
                sign = -sign
    return w, sign


def inspect_toy(circuit, a, b, coin):
    """The inspector's three checks: right answer, scratch wire back to 0, no stray sign."""
    w, sign = run_toy(circuit, a, b, coin)
    return w[3] == 1 - (a & b) and w[2] == 0 and sign == 1


WIRES = ['a', 'b', 'scratch', 'out']
TOY = [('CCX', 0, 1, 2), ('CX', 2, 3), ('HMR', 2), ('CZ_IF_M', 0, 1), ('X', 3)]  # out = NOT (a AND b)
TOY_BAD = [g for g in TOY if g[0] != 'CZ_IF_M']  # the planted mistake: the phase fix is gone

fig = plt.figure(figsize=(11.2, 3.3))
ax = fig.add_axes([0.0, 0.0, 0.62, 1.0])
xs = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0}
for i, name in enumerate(WIRES):
    ax.plot([0.4, 5.6], [3 - i, 3 - i], color=INK, lw=1.2, zorder=1)
    ax.text(0.3, 3 - i, name, ha='right', va='center', fontsize=11)
for x_, (gate, *q) in zip(xs.values(), TOY):
    ys = [3 - j for j in q]
    if gate in ('CCX', 'CX'):
        ax.plot([x_, x_], [min(ys), max(ys)], color=INK, lw=1.2, zorder=2)
        for yy in ys[:-1]:
            ax.scatter(x_, yy, s=40, color=INK, zorder=3)
        ax.scatter(x_, ys[-1], s=260, facecolors='white', edgecolors=ACCENT if gate == 'CCX' else INK, lw=1.6, zorder=3)
        ax.text(x_, ys[-1], '+', ha='center', va='center', fontsize=13, color=ACCENT if gate == 'CCX' else INK, zorder=4)
    elif gate == 'CZ_IF_M':
        ax.plot([x_, x_], [min(ys), max(ys)], color=INK, lw=1.2, zorder=2)
        for yy in ys:
            ax.scatter(x_, yy, s=40, color=INK, zorder=3)
        ax.plot([x_ - 1.0, x_ - 1.0, x_, x_], [1.24, 1.5, 1.5, min(ys)], color=MUTED, lw=1, ls='--')
        ax.text(x_ - 0.5, 1.56, 'm', fontsize=9, color=MUTED, ha='center')
        ax.text(x_ + 0.08, 2.5, 'CZ if m', fontsize=9, color=MUTED)
    else:
        ax.add_patch(plt.Rectangle((x_ - 0.28, ys[0] - 0.24), 0.56, 0.48, facecolor='white', edgecolor=INK, zorder=3))
        ax.text(x_, ys[0], gate, ha='center', va='center', fontsize=9.5, zorder=4)
ax.text(1.2, 1.3, 'one stamp (Toffoli)', ha='left', fontsize=9.5, color=ACCENT)
ax.text(3.0, 0.62, 'erase scratch: free', ha='center', fontsize=9.5, color=MUTED)
ax.set_xlim(-0.3, 5.9)
ax.set_ylim(-0.6, 3.5)
ax.axis('off')
ax2 = fig.add_axes([0.66, 0.08, 0.33, 0.84])
ax2.axis('off')
table = [['a', 'b', 'out', 'scratch', 'sign']]
for a in (0, 1):
    for b in (0, 1):
        outs = {tuple(run_toy(TOY, a, b, lambda m=m: m)[0][2:]) + (run_toy(TOY, a, b, lambda m=m: m)[1],) for m in (0, 1)}
        assert len(outs) == 1  # the same result whatever the measurement says
        (o,) = outs
        table.append([a, b, o[1], o[0], '+' if o[2] == 1 else '-'])
tab = ax2.table(cellText=table, loc='center', cellLoc='center')
tab.scale(1, 1.5)
for (r_, c_), cell in tab.get_celld().items():
    cell.set_edgecolor(GRID)
    if r_ == 0:
        cell.set_text_props(fontweight='bold')
ax2.set_title('truth table, run live: out = NOT (a AND b)', fontsize=11)
fig.suptitle('Figure 1. A circuit is sheet music: 4 seats, 1 stamp', x=0.0, ha='left', fontsize=12.5, fontweight='semibold')
show(fig, 'f1_toy_circuit.png', 'this notebook (run_toy); gate classes from the hosts\' paper, section II.A')

# CELL 12
def caught_after(circuit, salt, max_shots=60):
    seed = int.from_bytes(hashlib.sha256(repr(circuit).encode() + salt.encode()).digest()[:8], 'big')
    rng = random.Random(seed)  # the tests are a function of the file's hash and the salt
    for shot in range(1, max_shots + 1):
        a, b = rng.getrandbits(1), rng.getrandbits(1)
        if not inspect_toy(circuit, a, b, lambda: rng.getrandbits(1)):
            return shot
    return None


salts = [f'salt-{i}' for i in range(400)]
good = [caught_after(TOY, s) for s in salts]
bad = [caught_after(TOY_BAD, s) for s in salts]
assert all(g is None for g in good)  # the correct toy passes every test under every salt
n = np.arange(1, 41)
share = [np.mean([b_ is not None and b_ <= k for b_ in bad]) for k in n]
fig, ax = plt.subplots(figsize=(9.6, 3.8))
ax.step(n, share, where='post', color=ACCENT, lw=2, label='broken toy: share of 400 salts that caught it')
ax.plot(n, 1 - (7 / 8) ** n, color=INK, lw=1, ls='--', label='1 - (7/8)^shots')
ax.axhline(0, color=SECOND, lw=2, label='correct toy: caught under no salt')
k50 = int(n[np.argmax(np.array(share) >= 0.5)])
label_point(ax, k50, share[k50 - 1], f'half caught by shot {k50}')
ax.set_xlabel('shots (random test inputs)')
ax.set_ylabel('share of salts that exposed the mistake')
ax.set_ylim(-0.04, 1.04)
ax.legend(loc='lower right')
ax.set_title('Figure 2. Catching a planted mistake')
show(fig, 'f2_planted_mistake.png', 'this notebook (inspect_toy, 400 salts, seeds from SHA-256 of the circuit text plus salt)')

# CELL 14
data = bytearray((SUB / 'invsqrt.kmb').read_bytes())
print('invsqrt.kmb as pinned  ', hashlib.sha256(data).hexdigest())
data[len(data) // 2] ^= 1
print('one bit flipped        ', hashlib.sha256(data).hexdigest())
planted = SCRATCH / 'planted' / 'invsqrt.kmb'
planted.parent.mkdir(parents=True, exist_ok=True)
planted.write_bytes(data)
try:
    assert sha256(planted) == PINS['invsqrt.kmb']['sha256'], 'the pin refuses the flipped file'
    print('GREEN: the check did not notice (this must not happen)')
except AssertionError as err:
    print('RED, as it should be:', err)

# CELL 16
def gain(f, s):
    return s * (f ** (-1 / 3) - 1)  # the score is a geometric mean of three ratios


price = [
    ('fusion: one seat less', gain((ri['num_qubits'] - 1) / ri['num_qubits'], SCORE)),
    ('fusion: 1% fewer stamps', gain(0.99, SCORE)),
    ('message repair: one seat less', gain(((rd['num_qubits'] - 1) / rd['num_qubits']) ** 2, SCORE)),
    ('message repair: 1% fewer stamps', gain(0.99, SCORE)),
    ('chemistry: one seat less', gain((rc['num_qubits'] - 1) / rc['num_qubits'], SCORE)),
    ('chemistry: 1% fewer stamps', gain(0.99, SCORE)),
]
fig, ax = plt.subplots(figsize=(9.6, 3.6))
y = np.arange(len(price))[::-1]
for yi, (name, pts) in zip(y, price):
    seat = 'seat' in name
    ax.barh(yi, pts, height=0.6, color=ACCENT if seat else '#adb5bd')
    ax.text(pts + 0.002, yi, f'+{pts:.4f}', va='center', fontsize=10, color=ACCENT if seat else INK)
ax.set_yticks(y, [p[0] for p in price])
ax.grid(axis='y', visible=False)
ax.set_xlim(0, max(p[1] for p in price) * 1.25)
ax.set_xlabel(f'LB points gained, at this run\'s score {SCORE:.5f}')
ax.set_title('Figure 3. The price list: what one seat and 1% of the stamps are worth')
show(fig, 'f3_price_list.png', "this run's receipts (Q per job) and the geometric-mean formula of the hosts' metric code")

# CELL 19
relay = REC['relay']
fig, ax = plt.subplots(figsize=(10.2, 4.8))
qs = np.linspace(60, 87, 200)
for p in (relay[0], relay[5], relay[-1]):
    ax.plot(qs, p['cost'] / qs, color=GRID, lw=1.2, zorder=0)
colors = {'busyaprime': ACCENT, 'mrmeows': '#495057', 'Phuc Thai': '#495057', 'Dmitrii Gluzdov': '#495057'}
ax.plot([p['q'] for p in relay], [p['c'] for p in relay], color='#ced4da', lw=1, zorder=1)
for p in relay:
    ax.scatter(p['q'], p['c'], s=70 if p['who'] == 'busyaprime' else 45, color=colors[p['who']], zorder=3)
done = set()
for p in relay:
    if p['who'] not in done:
        ax.annotate(p['who'], (p['q'], p['c']), xytext=(4, 9), textcoords='offset points', fontsize=10,
                    color=colors[p['who']], fontweight='semibold' if p['who'] == 'busyaprime' else 'normal')
        done.add(p['who'])
end = relay[-1]
label_point(ax, end['q'], end['c'], f"{end['q']} seats, in this zip ({ratio['invsqrt']:.2f}x)", color=INK)
ax.set_xlim(87, 60)
ax.set_ylim(min(p['c'] for p in relay) - 45, max(p['c'] for p in relay) + 70)
ax.set_xlabel('seats (qubits), fewer to the right')
ax.set_ylabel('stamps per shot (Toffoli-equivalents)')
ax.set_title(f"Figure 4. The relay: {relay[0]['q']} to {end['q']} seats, {len({p['who'] for p in relay})} authors, "
             "one public line")
show(fig, 'f4_relay.png', 'data/qcc/circuits/invsqrt_*.verify.json and data/qcc/scout/regen/exp222/verify.json '
     "(our full verifier runs of each file); the file in this zip: our hosts'-metric runs of its bytes; "
     'grey curves: equal cost Q x C')

# CELL 22
sys.path += [KICKMIX_DIR, VERIFIERS_DIR]
from dqi import verify_submission_dqi as vdqi

n_shots = REC['ref']['dqi']['shots']
ks = np.arange(0, 7)
rk = [vdqi.repetition_overhead(vdqi.clopper_pearson_upper_bound(int(k), n_shots)) for k in ks]
assert all(abs(a - REC['rk'][str(k)]) < 1e-12 for k, a in zip(ks, rk))  # the same R(k) as at build time
counts = [REC['member_ks'].count(int(k)) for k in ks]
fig, ax = plt.subplots(figsize=(9.6, 3.8))
ax.bar(ks, rk, width=0.6, color=['#adb5bd'] * len(ks))
ax.bar([k_run], [rk[k_run]], width=0.6, color=ACCENT)
for k, r_, c_ in zip(ks, rk, counts):
    ax.text(k, r_ + 0.0012, f'{r_:.5f}' + (f'\n{c_} of {len(REC["member_ks"])} runs' if c_ else ''), ha='center',
            va='bottom', fontsize=8.5, color=INK)
ax.set_ylim(min(rk) - 0.01, max(rk) + 0.012)
ax.set_xticks(ks)
ax.set_xlabel(f'failed shots k, out of {n_shots:,}')
ax.set_ylabel('surcharge R(k)')
ax.set_title(f'Figure 5. The price of a mistake: this run failed {k_run} (orange)')
show(fig, 'f5_price_of_a_mistake.png', 'repetition_overhead and clopper_pearson_upper_bound imported live from '
     "verifiers/dqi/verify_submission_dqi.py; run counts: our Kaggle runs of this file, "
     "data/qcc/work/dqi/r24/kernel/prof1/output/r24/verify_salt*.json")

# CELL 24
pts = REC['dqi_points']
fig, ax = plt.subplots(figsize=(10.2, 4.8))
qq = np.linspace(855, 940, 200)
for cost in (4.4e11, 6.0e11, 8.0e11, 1.1e12):
    ax.plot(qq, cost / (REC['rk']['0'] * qq ** 2) / 1e3, color=GRID, lw=1.1, zorder=0)
    ax.text(858, cost / (REC['rk']['0'] * 858 ** 2) / 1e3, f'{cost:.1e}', fontsize=8, color=MUTED, ha='right', va='bottom')
pub = [p for p in pts if not p['ours']]
ours = [p for p in pts if p['ours']]
OFFSET = {'q919': (6, 4), 'sigma pair': (6, 5), 'v7': (6, 6), 'R5-A': (-44, 8), 'R12': (8, 0),
          'R18': (8, -12), 'SomeJoint gap': (6, -12), 'ViberTeam 881': (6, 5), 'DePie EXP-250a': (6, -13),
          'Ahmed V6': (-62, -12), 'mrmeows Q870': (8, 2), 'Ahmed V9 877 Q': (8, -4)}
ax.scatter([p['q'] for p in pub], [p['t'] / 1e3 for p in pub], s=40, color='#868e96', zorder=2)
zp = next(p for p in pts if p.get('in_zip'))
ax.scatter([zp['q']], [zp['t'] / 1e3], s=150, facecolors='none', edgecolors=INK, lw=1.8, zorder=4)
ax.plot([p['q'] for p in ours], [p['t'] / 1e3 for p in ours], '-o', color=ACCENT, ms=5, lw=1.4, zorder=3)
for p in pts:
    if p['label'] in OFFSET:
        text = p['label'] + (f", in this zip ({ratio['dqi']:.2f}x)" if p.get('in_zip') else '')
        text += f", ours, best ({REC['own_hist']['dqi']:.2f}x)" if p['label'] == 'R18' else ''
        ax.annotate(text, (p['q'], p['t'] / 1e3), xytext=OFFSET[p['label']], textcoords='offset points', fontsize=8.5,
                    color=ACCENT if p['ours'] else (INK if p.get('in_zip') else '#868e96'),
                    fontweight='semibold' if p['label'] == 'R18' or p.get('in_zip') else 'normal')
ref_ = REC['ref']['dqi']
ax.text(0.99, 0.97, f"Google's reference, far off this chart: {ref_['q']:,} seats, {ref_['t'] / 1e3:,.0f}k stamps",
        transform=ax.transAxes, ha='right', va='top', fontsize=9, color=INK)
ax.set_xlim(940, 855)
ax.set_ylim(min(p['t'] for p in pts) / 1e3 - 60, max(p['t'] for p in pts) / 1e3 + 80)
ax.set_xlabel('seats (qubits), fewer to the right')
ax.set_ylabel('stamps per shot, thousands (mean)')
ax.set_title("Figure 6. The decoder's diet: our line (orange), public files we verified (grey), this zip (circled)")
show(fig, 'f6_decoder_diet.png', 'data/qcc/circuits/dqi_*.json and data/qcc/work/dqi/r24/kernel/prof1/output/r24/verify_salt*.json '
     '(every run of each file); grey curves: equal cost R(0) x Q^2 x T')

# CELL 26
parts = [('volume, lambda_eff', ref['dfthcsos']['lam'], rc['lambda_eff']),
         ('stamps, T', ref['dfthcsos']['t'], rc['toffoli_count']),
         ('seats, Q', ref['dfthcsos']['q'], rc['num_qubits'])]
fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.0))
for ax, (name, g, o) in zip(axes, parts):
    ax.bar([0, 1], [g, o], color=['#adb5bd', ACCENT], width=0.6)
    ax.set_xticks([0, 1], ['Google', 'ours'])
    ax.set_title(f'{name}: {g / o:.2f}x', fontsize=11)
    for x_, v in ((0, g), (1, o)):
        ax.text(x_, v, f'{v:,.2f}' if v < 100 else f'{v:,.0f}', ha='center', va='bottom', fontsize=9.5)
    ax.grid(axis='x', visible=False)
    ax.set_ylim(0, g * 1.18)
fig.suptitle(f"Figure 7. The chemistry bill, factor by factor: {ratio['dfthcsos']:.2f}x in all", x=0.0, ha='left',
             fontsize=12.5, fontweight='semibold')
fig.tight_layout()
show(fig, 'f7_chemistry_bill.png', "/kaggle/working/receipts/dfthcsos.json (this run); data/qcc/runs/sample_dfthcsos.json")

# CELL 28
c3 = REC['chem3']
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.2))
names = [c['who'] for c in c3]
cols = ['#adb5bd', '#868e96', ACCENT]
lam_live = R['dfthcsos']['lambda_eff']
assert abs(lam_live - c3[-1]['lam']) < 1e-12  # this run's verifier reads the same volume as our record
ax.barh(range(3), [c['lam'] for c in c3], color=cols, height=0.6)
for i, c in enumerate(c3):
    ax.text(c['lam'] + 0.0008, i, f"{c['lam']:.6f}", va='center', fontsize=9.5)
ax.set_xlim(min(c['lam'] for c in c3) - 0.01, max(c['lam'] for c in c3) + 0.012)
ax.set_yticks(range(3), names)
ax.invert_yaxis()
ax.grid(axis='y', visible=False)
ax.set_title('volume knob, lambda_eff', fontsize=11)
base = c3[0]['cost']
ax2.barh(range(3), [100 * (c['cost'] / base - 1) for c in c3], color=cols, height=0.6)
for i, c in enumerate(c3):
    d = 100 * (c['cost'] / base - 1)
    ax2.text(d - 0.001, i, f'{d:+.4f}%', va='center', ha='right' if d < 0 else 'left', fontsize=9.5)
ax2.set_yticks(range(3), ['', '', ''])
ax2.invert_yaxis()
ax2.axvline(0, color=MUTED, lw=0.8)
ax2.set_xlim(-0.045, 0.012)
ax2.grid(axis='y', visible=False)
ax2.set_title("chemistry cost against Ahmed Berat Ozer's V9", fontsize=11)
fig.suptitle('Figure 7b. Same 15 circuits, three weights files', x=0.0, ha='left', fontsize=12.5, fontweight='semibold')
fig.tight_layout()
show(fig, 'f7b_three_weights.png', 'our verifier runs of each weights file (sources above); this run: /kaggle/working/receipts/dfthcsos.json')

# CELL 30
cal = REC['calib']
fig, ax = plt.subplots(figsize=(10.2, 3.9))
for i, c in enumerate(cal):
    lo, hi = 100 * (c['lo'] / c['hi'] - 1), 0.0
    ax.plot([i, i], [lo, hi], color='#adb5bd', lw=7, solid_capstyle='butt', zorder=1)
    d = 100 * (c['lb'] / c['hi'] - 1)
    inside = lo - 2e-3 <= d <= 2e-3
    ax.scatter(i, d, s=60, color=ACCENT if c['name'] == REC['bundle'] else (INK if inside else SECOND), zorder=3)
    if not inside:
        ax.annotate(f"{c['name']}: LB {c['lb']:.5f}", (i, d), xytext=(6, -4), textcoords='offset points', fontsize=9,
                    color=SECOND)
ax.set_xticks(range(len(cal)), [c['name'].replace('sub_', '') for c in cal])
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_ylabel('LB vs our best-run prediction, %')
ax.set_xlabel('scored submission (grey bar: the range our own runs predicted)')
ax.set_title("Figure 8. Our judge and Kaggle's judge")
show(fig, 'f8_judges.png', f"submissions list and data/qcc/submissions/sub_*.json (predicted_score_worst_runs, _best_runs)")

# CELL 32
led = REC['ledger']
fig, ax = plt.subplots(figsize=(9.6, 2.6))
y = np.arange(3)[::-1]
for yi, t in zip(y, TASKS):
    ax.barh(yi, led[t]['failed'], color=MUTED, height=0.55)
    ax.barh(yi, led[t]['other'], left=led[t]['failed'], color=ACCENT, height=0.55)
    ax.text(led[t]['rows'] + 2, yi, f"{led[t]['rows']} rows", va='center', fontsize=10)
ax.set_yticks(y, [JOB[t] for t in TASKS])
ax.grid(axis='y', visible=False)
ax.set_xlim(0, max(v['rows'] for v in led.values()) * 1.15)
ax.set_xlabel('ledger rows (grey: the verdict names a failure; orange: kept, measured or superseded)')
ax.set_title(f"Figure 9. {sum(v['rows'] for v in led.values())} attempts in the ledgers")
show(fig, 'f9_attempts.png', 'competitions/33_qcc/{invsqrt,dqi,dfthcsos}/33_attempt_ledger.jsonl')

# CELL 35
import datetime as dt
tide = REC['tide']
ts = [dt.datetime.fromisoformat(x['t']) for x in tide]
fig, ax = plt.subplots(figsize=(10.4, 4.4))
for key, name, color in (('top1', 'rank 1', INK), ('gold', 'gold line (rank 12)', '#495057'),
                         ('r100', 'rank 100', '#868e96'), ('median', 'median', '#adb5bd')):
    xs_ = [t for t, x in zip(ts, tide) if x[key]]
    ys_ = [x[key] for x in tide if x[key]]
    ax.plot(xs_, ys_, color=color, lw=1.8, label=name)
for a, b in REC['tide_gaps']:
    ax.axvspan(dt.datetime.fromisoformat(a), dt.datetime.fromisoformat(b), color=GRID, alpha=0.6, lw=0)
    ax.text(dt.datetime.fromisoformat(a), 0.6, ' no board saved', fontsize=8.5, color=MUTED)
steps = REC['ours_steps']
sx = [dt.datetime.fromisoformat(s['t']) for s in steps] + [ts[-1]]
sy = [s['score'] for s in steps] + [steps[-1]['score']]
ax.step(sx, sy, where='post', color=ACCENT, lw=2.2, label='ours, best scored')
own = next(s for s in steps if s['name'] == REC['own_hist']['bundle'])
label_point(ax, dt.datetime.fromisoformat(own['t']), own['score'], f"{own['score']:.5f}, our own circuits")
ax.scatter([ts[-1]], [SCORE], marker='*', s=160, color=ACCENT, zorder=5, label='this zip, this run')
ax.set_ylim(0, max(x['top1'] for x in tide) * 1.08)
ax.set_ylabel('public LB score')
ax.legend(loc='upper left', ncol=3)
ax.set_title('Figure 10. The tide')
fig.autofmt_xdate()
show(fig, 'f10_tide.png', 'data/qcc/lb/**/*publicleaderboard-*.csv (every saved board); our submissions list')

# CELL 36
scores = np.array(REC['board_scores'])
vals, cnt = np.unique(np.round(scores, 5), return_counts=True)
fig, ax = plt.subplots(figsize=(10.4, 4.0))
keep = vals >= 15
ax.vlines(vals[keep], 0, cnt[keep], color='#868e96', lw=1.6)
for i, c in enumerate(sorted(REC['clusters'], key=lambda c: -c['n'])[:3]):
    ax.annotate(f"{c['n']} teams at {c['score']:.5f}: {c['notebook'].split('/')[0]}'s notebook", (c['score'], c['n']),
                xytext=(0.36, 0.93 - 0.1 * i), textcoords='axes fraction', ha='left', va='center', fontsize=8.5,
                color=INK, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
ax.axvline(SCORE, color=ACCENT, lw=1.6)
ax.text(SCORE, max(cnt) * 0.45, f" this zip {SCORE:.5f}", color=ACCENT, fontsize=10)
ax.set_xlabel('score (rows at or above 15)')
ax.set_ylabel('teams with exactly this score')
ax.set_ylim(0, max(cnt) * 1.25)
ax.set_title(f'Figure 11. The tie wall: {len(scores):,} teams, {int((cnt >= 2).sum())} scores shared by two or more')
show(fig, 'f11_tie_wall.png', 'the latest saved board; notebook names: data/qcc/scout/2026-10-27T1609/33_clusters.txt')

# CELL 39
rows = ['| job | verdict | seats | stamps per shot | failed shots | cost, this run | cost, record | rel. difference | seconds |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for t in TASKS:
    r = R[t]
    stamps = r.get('toffoli_equivalent', r['toffoli_count'])
    fails = r.get('value_errors', 0) + r.get('phase_errors', 0) if t == 'dqi' else 0
    rows.append(f"| {JOB[t]} | **{r['status']}** | {r['num_qubits']} | {stamps:,.2f} | {fails} | {r['score']:,.4f} | "
                f"{EXP['costs'][t]:,.4f} | {REL[t]:.1e} | {r['_seconds']:.0f} |")
rows.append(f"| **score** | | | | | **{SCORE:.5f}** | {EXP['score']:.5f} | {REL_SCORE:.1e} | {RESULT['wall_seconds']:.0f} |")
display(Markdown('\n'.join(rows) + f"\n\n`submission.zip`: {OUT_ZIP.stat().st_size:,} B, SHA-256 `{ZIP_SHA}`; "
                 f"same bytes as our submission {REC['bundle']}: **{SAME}**. Record: `{EXP['metric_record']}`."))

# CELL 40
WALL = time.monotonic() - T0
GATES = {f"G2 bytes equal {REC['bundle']}": SAME,
         'G3 every cost within 1e-9 of the record': max(REL.values()) <= 1e-9 and REL_SCORE <= 1e-9,
         'G4 wall time at most 1,800 s': WALL <= 1800,
         'G4 peak RSS at most 24 GiB': RSS_GIB <= 24}
summary = {'bundle': REC['bundle'], 'score': SCORE, 'expected_score': EXP['score'], 'rel_score': REL_SCORE,
           'costs': {t: R[t]['score'] for t in TASKS}, 'rel': REL, 'k_dqi': k_run, 'zip_sha256': ZIP_SHA,
           'same_bytes': SAME, 'wall_seconds': WALL, 'pin_seconds': PIN_SECONDS, 'zip_seconds': ZIP_SECONDS,
           'metric_seconds': RESULT['wall_seconds'], 'verifier_seconds': {t: R[t]['_seconds'] for t in TASKS},
           'peak_rss_gib': RSS_GIB, 'maxrss_runner_kib': RESULT['maxrss_self_kib'], 'cpus': os.cpu_count(),
           'python': platform.python_version(), 'metric_sha256': METRIC_SHA, 'inputs': FOUND, 'gates': GATES,
           'own_lane': {t: r['score'] for t, r in OWN_REPORTS.items()}, 'start_utc': START_UTC}
(WORKING / 'qcc33_v8_run.json').write_text(json.dumps(summary, indent=1, default=float))
for name, ok in GATES.items():
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
print(f'wall {WALL:,.0f} s, peak RSS {RSS_GIB:.2f} GiB, score {SCORE:.5f}')
assert all(GATES.values()), GATES