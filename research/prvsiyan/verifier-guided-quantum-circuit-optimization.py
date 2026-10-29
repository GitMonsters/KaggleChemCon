# CELL 2
import matplotlib.pyplot as plt

TASKS = ["invsqrt", "dqi", "dfthcsos"]
COST = {"invsqrt": 103546.168164559, "dqi": 448601212111.76025, "dfthcsos": 376139673.7231457}
REFERENCE = {"invsqrt": 2076774.233715282, "dqi": 24439473914136.668, "dfthcsos": 3355599588.3155365}
fig, ax = plt.subplots(figsize=(8.4, 4.4))
x = range(len(TASKS)); width = 0.36
ax.bar([i-width/2 for i in x], [COST[t] for t in TASKS], width, label="Selected, verifier PASS", color="#147d92")
ax.bar([i+width/2 for i in x], [REFERENCE[t] for t in TASKS], width, label="Reference", color="#d7a84b")
ax.set_yscale("log"); ax.set_xticks(list(x), TASKS); ax.set_ylabel("Verifier cost (log scale; lower is better)")
ax.set_title("Measured task costs versus reference")
ax.grid(axis="y", which="both", alpha=.2); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False)
fig.tight_layout(); plt.show()


# CELL 3
import matplotlib.pyplot as plt

TASKS = ["invsqrt", "dqi", "dfthcsos"]
COST = {"invsqrt": 103546.168164559, "dqi": 448601212111.76025, "dfthcsos": 376139673.7231457}
REFERENCE = {"invsqrt": 2076774.233715282, "dqi": 24439473914136.668, "dfthcsos": 3355599588.3155365}
contribution = [(REFERENCE[t] / COST[t]) ** (1/3) for t in TASKS]
fig, ax = plt.subplots(figsize=(8.4, 4.2))
bars = ax.bar(TASKS, contribution, color=["#147d92", "#48a9a6", "#d7a84b"])
ax.axhline(1, color="#4b5563", linewidth=1, linestyle="--", label="Reference parity (1)")
ax.set_ylabel("Equal-weight factor: (reference / cost)^(1/3)")
ax.set_title("Each task’s multiplicative contribution to the geometric mean")
ax.grid(axis="y", alpha=.2); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False)
for bar, value in zip(bars, contribution): ax.text(bar.get_x()+bar.get_width()/2, value, f"{value:.3f}×", ha="center", va="bottom")
fig.tight_layout(); plt.show()


# CELL 5
import matplotlib.pyplot as plt

versions = [9, 10, 11, 12, 13]
# Local draft-validation metrics exist for v12 and v13 only; missing points stay missing.
local_metric = [float("nan"), float("nan"), float("nan"), 19.599978502664182, 21.36170359590384]
# Public score records: v12 completed; v13 is still pending and has no score yet.
public_score = [float("nan"), float("nan"), float("nan"), 19.57896, float("nan")]
fig, ax = plt.subplots(figsize=(8.4, 4.4))
ax.plot(versions, local_metric, marker="o", linewidth=2.2, color="#147d92", label="Local draft verifier metric")
ax.plot(versions, public_score, marker="s", linestyle="--", linewidth=1.8, color="#d7a84b", label="Completed Kaggle public score")
ax.set_xticks(versions, [f"v{v}" for v in versions]); ax.set_ylabel("Score (higher is better)")
ax.set_title("Available v9–v13 score evidence; blanks are unrecorded")
ax.grid(axis="y", alpha=.22); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False)
ax.annotate("v13 public score pending", (13, 21.36170359590384), xytext=(-100, -22), textcoords="offset points", color="#374151")
fig.tight_layout(); plt.show()


# CELL 8
import hashlib
import json
import math
import os
import pathlib
import shutil
import time
import zipfile

CONFIG = {'candidates': {'dfthcsos': [{'cost': 376139673.7231457,
                              'label': 'intended_current_public',
                              'member_sha256': {'dfthcsos/angle_k_0.kmb': '492d50adcbfde8d624fd277c83c30cf5c760093004dd314ba907b9daad984198',
                                                'dfthcsos/angle_k_1.kmb': 'a42e1d56ece1ddc9f3b4124515ba8ac4293f5bb1ff9e68948913da06df0692b9',
                                                'dfthcsos/angle_k_10.kmb': 'c12ac38d33d7b2193e706d49366e3bfe5a4ce69b1e9472477aa9acc4137ac735',
                                                'dfthcsos/angle_k_11.kmb': 'ae32f7d4f8557c03c02aa6bf1ec9600a967fc6f4313656d560b092394da33bf8',
                                                'dfthcsos/angle_k_12.kmb': '787de6b7249ffe92f3661aacb77237d21f0306f1c4e19fdbf76bdd3447aa4bb1',
                                                'dfthcsos/angle_k_13.kmb': '6847e1a4fdd1391f4e633a17da40983b01b264dfca886e576c9032c9bdaafce7',
                                                'dfthcsos/angle_k_14.kmb': '96b86e279652273cb65163ac9004fa049d0cb51b24aa07688de15f51ffaaea52',
                                                'dfthcsos/angle_k_2.kmb': '70576d21b697c24d149727126648f073096729aea686df21ce0a559e0d55acb8',
                                                'dfthcsos/angle_k_3.kmb': 'a6558e9a778296ea3edbbd9ff45453d98309d6749300422c8a43abc288d63d9a',
                                                'dfthcsos/angle_k_4.kmb': '9b7aad07ae9439cb32a277f70ea402c1bc03bb31b0c5db42fcc5c5ce2840a55b',
                                                'dfthcsos/angle_k_5.kmb': '1e702c1fbe5f39ab5037d9d7f5564241a1909dd1b8b558c1878673ba929b1aa0',
                                                'dfthcsos/angle_k_6.kmb': '36f6ca86b8099dbd98174f0d87acd4e5b48648c252e740983f03a6937f61bbdc',
                                                'dfthcsos/angle_k_7.kmb': '42131c214894fcc0501f37509a847cbd3e7a2c44ac9d2805d5f9690b00ac5257',
                                                'dfthcsos/angle_k_8.kmb': 'b11bc336b762fc0b803cd4d32c95f44df95236a172eb1d62ef02972b9d45c664',
                                                'dfthcsos/angle_k_9.kmb': '8b5ca34495a8e5c15063db2fd5029381eea1af586025ef4f66e99ee0c3a6e63d',
                                                'dfthcsos/submission.npz': '4a27caf6c2d22192bbfd3387ff2f1b8c9e6512821fdbc8e56015374cfe1e1050'},
                              'receipt_path': 'pulls/alexchilton-21-39/output/21-39-lb-verified-three-task-portfolio.log; '
                                              'exact member hashes cross-checked against '
                                              'q-merge-k/validation.json',
                              'source': 'alexchilton/21-39-lb-verified-three-task-portfolio',
                              'status': 'PASS',
                              'task': 'dfthcsos',
                              'verified_shots': 87424},
                             {'cost': 377049552.6117826,
                              'label': 'best_available_fallback',
                              'member_sha256': {'dfthcsos/angle_k_0.kmb': '8b8b41edf8f5dd9864b3208c2038ae66925c6149c8812233960a50684706b560',
                                                'dfthcsos/angle_k_1.kmb': 'fea1931c7c02ff09b96201162fb9c78edb0a1336b601ec94c3091e10ef06883d',
                                                'dfthcsos/angle_k_10.kmb': '7b16c7c360a312b44a13ef8175c80fb708cdf1a67abe4cd6154458ab7c7f98b5',
                                                'dfthcsos/angle_k_11.kmb': 'da02051a407b39846dfe1a4184ec877ea80ceb0cae10e4ba77f0d7e28dced5a8',
                                                'dfthcsos/angle_k_12.kmb': 'b6cae968da20232d5fb711b6bbe7b2f1f1a1a08f1120438f3c2e6dfd48404a7d',
                                                'dfthcsos/angle_k_13.kmb': 'f2eb89ec9348250c2446c523b33b1a659e43eebae2d33802192de2256c08f8f9',
                                                'dfthcsos/angle_k_14.kmb': 'cf66d59e0d4c7bac9cf3ce9a9c635d40a6ca8f13921dc4816d12919c13777855',
                                                'dfthcsos/angle_k_2.kmb': 'b0c0dc3e34a3b31193b1455211e8be123dc9ea390da4372f94c1473ac2a3d4a8',
                                                'dfthcsos/angle_k_3.kmb': '30983654a70f9d87586d5668f470aa63082389aaef03047da76ab124a5240bbe',
                                                'dfthcsos/angle_k_4.kmb': '8c42ca14958453b6d7f73f3d5563cd7ff60643d11bb1ad3b084583d66a61ea96',
                                                'dfthcsos/angle_k_5.kmb': '95db80da0cb1a0168d641b9c271d7809261b73e84f158fa0c9ef68f2a526e4a5',
                                                'dfthcsos/angle_k_6.kmb': '380fdf8d87886f1fd16a6e61fbc43ca635c086c80587c9c00417231d7f4ae4a5',
                                                'dfthcsos/angle_k_7.kmb': '29b1e13295c716c107b4a9f97415c52bfa3567116110353ac5928fc15721b955',
                                                'dfthcsos/angle_k_8.kmb': '57ea13f5a8dc0bfeb9d47752f6ce56f7a3e6ccdbbbf803fe02fbf51f8dfa9f32',
                                                'dfthcsos/angle_k_9.kmb': '33124e909c9a3e7e0fad0ce375eeb79785775103a787fddd45abab4e8ff79e61',
                                                'dfthcsos/submission.npz': '8229861b3ee22316e144c900737dfc54f7a30ea88de77ff5391f3375d050fa87'},
                              'receipt_path': 'qfam-dft-d/validation.json#pass_receipt',
                              'source': 'dmitriigluzdov/quantum-circuit-compilation-from-source',
                              'status': 'PASS',
                              'task': 'dfthcsos',
                              'verified_shots': 87424},
                             {'cost': 426973779.3811362,
                              'label': 'v12_fallback',
                              'member_sha256': {'dfthcsos/angle_k_0.kmb': 'b631c3cafd2edc600edeed997f28c1b1b8a71234189bcc7d6a3afdf4a47408db',
                                                'dfthcsos/angle_k_1.kmb': '974dc12398186f359e79d38d733720c5231686d7f591dcb1ac1b5d20be27f47e',
                                                'dfthcsos/angle_k_10.kmb': 'ac3cdb0d8d742d9629d3d5e3d5acb843012a084557587fe2c0ad9b9058da06d5',
                                                'dfthcsos/angle_k_11.kmb': '9b103b5aadfc52e1b481894dff426dbf7443b5a3530db2c1e325ef6a39ba63e2',
                                                'dfthcsos/angle_k_12.kmb': 'd872682a1672429fdbbb44f56dbac9f4a0e951231817926d5470ff36f64aac3d',
                                                'dfthcsos/angle_k_13.kmb': '42044bcb48d80947ef2900aea4eb2eaa8593f1b809d37bada48f1ab169b44ae3',
                                                'dfthcsos/angle_k_14.kmb': 'f3bb8ce79e39ecddad5bb1ff57e6818158c4cba1133b54f0009f1133b8585d49',
                                                'dfthcsos/angle_k_2.kmb': 'e25d78895511c50babe940b21314ede6ad79794d38a9da53efaefbdc802864ac',
                                                'dfthcsos/angle_k_3.kmb': 'ac59a43cf75a516937ad12900916c292add65558902df77b18d0600cc45a7ecb',
                                                'dfthcsos/angle_k_4.kmb': '64d9dfc71e0fb4c5c55e4a01ba9063c579d17d2602551a0899253f8edd1dc136',
                                                'dfthcsos/angle_k_5.kmb': '6c4b1e3a59e46a604528f80912f4ccfd9ce689c0f39cd548521d0663d9d1821a',
                                                'dfthcsos/angle_k_6.kmb': '9932019836bbe8217a8fd34d1b54bbaccf9591d1c83f56c2a0cb20e2e14a2c7d',
                                                'dfthcsos/angle_k_7.kmb': '7f1444666cf99cac8533089c862edfeb60db5e41daedf693c03af0d0be4b4392',
                                                'dfthcsos/angle_k_8.kmb': 'd748449d2c77cf858bdd90d69054bc3a3ac1d81ad2501c21d85e5b04a824e66b',
                                                'dfthcsos/angle_k_9.kmb': '039ec2a8c81ff647a83832dc76a687947a80f81202361b9af2f1c6cef1e29268',
                                                'dfthcsos/submission.npz': 'a5eabd76c2b2869e39511a53130cd909034dec6f26fa3db61354a9d12ffe6413'},
                              'receipt_path': 'q-pub-h/sources.json#fallback_receipt_candidates.dfthcsos[0]',
                              'source': 'lightningv08/bezout-schur-5-word-qrom',
                              'status': 'PASS',
                              'task': 'dfthcsos',
                              'verified_shots': 70144}],
                'dqi': [{'cost': 448601212111.76025,
                         'label': 'intended_current_public',
                         'member_sha256': {'dqi.kmb': '7beea291a5de36ef2b4fb50585b2f8d7cce02d3c575b491cf5dcc93e9d9655f6'},
                         'receipt_path': 'pulls/alexchilton-21-39/output/21-39-lb-verified-three-task-portfolio.log; '
                                         'exact member hashes cross-checked against '
                                         'q-merge-k/validation.json',
                         'source': 'alexchilton/21-39-lb-verified-three-task-portfolio',
                         'status': 'PASS',
                         'task': 'dqi',
                         'verified_shots': 102400},
                        {'cost': 487993123590.66656,
                         'label': 'v12_fallback',
                         'member_sha256': {'dqi.kmb': '569863ae0fafd5f94fe91b498ec67ed0a4d96b826ced31cb0ff57c7fe6c063cb'},
                         'receipt_path': 'draft-v12/validation.json#per_task.dqi',
                         'source': 'lightningv08/bezout-schur-5-word-qrom',
                         'status': 'PASS',
                         'task': 'dqi',
                         'verified_shots': 102400}],
                'invsqrt': [{'cost': 103546.168164559,
                             'label': 'intended_current_public',
                             'member_sha256': {'invsqrt.kmb': '0766a111a76f241aa57fbbed466436d743be13436943ddb0628c291ca07d67d6'},
                             'receipt_path': 'pulls/alexchilton-21-39/output/21-39-lb-verified-three-task-portfolio.log; '
                                             'exact member hashes cross-checked against '
                                             'q-merge-k/validation.json',
                             'source': 'alexchilton/21-39-lb-verified-three-task-portfolio',
                             'status': 'PASS',
                             'task': 'invsqrt',
                             'verified_shots': 200540928},
                            {'cost': 108560.35824641786,
                             'label': 'v12_fallback',
                             'member_sha256': {'invsqrt.kmb': '9ac50386192a544f353ab0f46604e5e072981b8bfc8521e8551c689a9445e3b9'},
                             'receipt_path': 'draft-v12/validation.json#per_task.invsqrt',
                             'source': 'lightningv08/bezout-schur-5-word-qrom',
                             'status': 'PASS',
                             'task': 'invsqrt',
                             'verified_shots': 200540928}]},
 'projected_score_current_alex_receipts': 21.36170359590384,
 'projected_score_qmerge_receipts': 21.36169824615438,
 'reference_costs': {'dfthcsos': 3355599588.3155365,
                     'dqi': 24439473914136.668,
                     'invsqrt': 2076774.233715282},
 'required_members': ['invsqrt.kmb',
                      'dqi.kmb',
                      'dfthcsos/angle_k_0.kmb',
                      'dfthcsos/angle_k_1.kmb',
                      'dfthcsos/angle_k_2.kmb',
                      'dfthcsos/angle_k_3.kmb',
                      'dfthcsos/angle_k_4.kmb',
                      'dfthcsos/angle_k_5.kmb',
                      'dfthcsos/angle_k_6.kmb',
                      'dfthcsos/angle_k_7.kmb',
                      'dfthcsos/angle_k_8.kmb',
                      'dfthcsos/angle_k_9.kmb',
                      'dfthcsos/angle_k_10.kmb',
                      'dfthcsos/angle_k_11.kmb',
                      'dfthcsos/angle_k_12.kmb',
                      'dfthcsos/angle_k_13.kmb',
                      'dfthcsos/angle_k_14.kmb',
                      'dfthcsos/submission.npz'],
 'target_member_hashes': {'dfthcsos': {'dfthcsos/angle_k_0.kmb': '492d50adcbfde8d624fd277c83c30cf5c760093004dd314ba907b9daad984198',
                                       'dfthcsos/angle_k_1.kmb': 'a42e1d56ece1ddc9f3b4124515ba8ac4293f5bb1ff9e68948913da06df0692b9',
                                       'dfthcsos/angle_k_10.kmb': 'c12ac38d33d7b2193e706d49366e3bfe5a4ce69b1e9472477aa9acc4137ac735',
                                       'dfthcsos/angle_k_11.kmb': 'ae32f7d4f8557c03c02aa6bf1ec9600a967fc6f4313656d560b092394da33bf8',
                                       'dfthcsos/angle_k_12.kmb': '787de6b7249ffe92f3661aacb77237d21f0306f1c4e19fdbf76bdd3447aa4bb1',
                                       'dfthcsos/angle_k_13.kmb': '6847e1a4fdd1391f4e633a17da40983b01b264dfca886e576c9032c9bdaafce7',
                                       'dfthcsos/angle_k_14.kmb': '96b86e279652273cb65163ac9004fa049d0cb51b24aa07688de15f51ffaaea52',
                                       'dfthcsos/angle_k_2.kmb': '70576d21b697c24d149727126648f073096729aea686df21ce0a559e0d55acb8',
                                       'dfthcsos/angle_k_3.kmb': 'a6558e9a778296ea3edbbd9ff45453d98309d6749300422c8a43abc288d63d9a',
                                       'dfthcsos/angle_k_4.kmb': '9b7aad07ae9439cb32a277f70ea402c1bc03bb31b0c5db42fcc5c5ce2840a55b',
                                       'dfthcsos/angle_k_5.kmb': '1e702c1fbe5f39ab5037d9d7f5564241a1909dd1b8b558c1878673ba929b1aa0',
                                       'dfthcsos/angle_k_6.kmb': '36f6ca86b8099dbd98174f0d87acd4e5b48648c252e740983f03a6937f61bbdc',
                                       'dfthcsos/angle_k_7.kmb': '42131c214894fcc0501f37509a847cbd3e7a2c44ac9d2805d5f9690b00ac5257',
                                       'dfthcsos/angle_k_8.kmb': 'b11bc336b762fc0b803cd4d32c95f44df95236a172eb1d62ef02972b9d45c664',
                                       'dfthcsos/angle_k_9.kmb': '8b5ca34495a8e5c15063db2fd5029381eea1af586025ef4f66e99ee0c3a6e63d',
                                       'dfthcsos/submission.npz': '4a27caf6c2d22192bbfd3387ff2f1b8c9e6512821fdbc8e56015374cfe1e1050'},
                          'dqi': {'dqi.kmb': '7beea291a5de36ef2b4fb50585b2f8d7cce02d3c575b491cf5dcc93e9d9655f6'},
                          'invsqrt': {'invsqrt.kmb': '0766a111a76f241aa57fbbed466436d743be13436943ddb0628c291ca07d67d6'}}}
TASKS = ("invsqrt", "dqi", "dfthcsos")
INPUT_ROOT = pathlib.Path(os.environ.get("VQCC_INPUT_ROOT", "/kaggle/input"))
OUTPUT_ROOT = pathlib.Path(os.environ.get("VQCC_OUTPUT_ROOT", "/kaggle/working"))
STARTED = time.perf_counter()


def digest_stream(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1 << 20), b""):
        h.update(chunk)
    return h.hexdigest()


def relative_source_group(path):
    try:
        parts = path.relative_to(INPUT_ROOT).parts
    except Exception:
        parts = path.parts
    if not parts:
        return str(INPUT_ROOT)
    # Kaggle may mount a competition under /kaggle/input/competitions/<slug>.
    if parts[0] == "competitions" and len(parts) > 1:
        return str(INPUT_ROOT.joinpath(*parts[:2]))
    return str(INPUT_ROOT / parts[0])


def enumerate_inputs():
    paths = []
    print("Recursive input audit root:", INPUT_ROOT)
    if not INPUT_ROOT.exists():
        print("[missing-root]", INPUT_ROOT)
        return paths
    print("[dir]", INPUT_ROOT)
    try:
        paths = sorted(INPUT_ROOT.rglob("*"), key=lambda p: str(p))
    except BaseException as exc:
        print("[walk-error]", type(exc).__name__, str(exc))
    for path in paths:
        try:
            print("[dir]" if path.is_dir() else "[file]", path)
        except BaseException as exc:
            print("[path-error]", path, type(exc).__name__, str(exc))
    return [p for p in paths if p.is_file()]


def zip_candidate(path, candidate):
    try:
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if not n.endswith("/")]
            references = {}
            for expected, expected_sha in candidate["member_sha256"].items():
                tail = expected.split("/")[-1]
                matches = [n for n in names if n == expected or n.endswith("/" + expected)]
                if not matches:
                    matches = [n for n in names if n.split("/")[-1] == tail]
                if len(matches) != 1:
                    return None
                name = matches[0]
                with archive.open(name) as stream:
                    actual_sha = digest_stream(stream)
                if actual_sha != expected_sha:
                    return None
                references[expected] = {"kind": "zip", "path": str(path), "member": name, "sha256": actual_sha}
            return references
    except BaseException as exc:
        print("[unreadable-zip]", path, type(exc).__name__, str(exc))
        return None


def direct_candidate(group_paths, candidate):
    references = {}
    for expected, expected_sha in candidate["member_sha256"].items():
        tail = expected.split("/")[-1]
        matches = [p for p in group_paths if p.name == tail]
        if expected.startswith("dfthcsos/"):
            matches = [p for p in matches if "dfthcsos" in p.parts]
        good = []
        for path in matches:
            try:
                with path.open("rb") as stream:
                    if digest_stream(stream) == expected_sha:
                        good.append(path)
            except BaseException:
                continue
        if len(good) != 1:
            return None
        references[expected] = {"kind": "file", "path": str(good[0]), "sha256": expected_sha}
    return references


def find_candidate(candidate, archives, direct_groups):
    for path in archives:
        found = zip_candidate(path, candidate)
        if found is not None:
            return found
    for group_paths in direct_groups.values():
        found = direct_candidate(group_paths, candidate)
        if found is not None:
            return found
    return None


def copy_reference(zout, target_name, reference):
    with zout.open(target_name, "w") as target:
        if reference["kind"] == "zip":
            with zipfile.ZipFile(reference["path"]) as source_zip:
                with source_zip.open(reference["member"]) as source:
                    shutil.copyfileobj(source, target, length=1 << 20)
        else:
            with open(reference["path"], "rb") as source:
                shutil.copyfileobj(source, target, length=1 << 20)


def write_figures(selected, metric):
    figure_root = OUTPUT_ROOT / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    refs = CONFIG["reference_costs"]
    target_costs = {task: CONFIG["candidates"][task][0]["cost"] for task in TASKS}
    picked_costs = {task: selected[task]["candidate"]["cost"] if task in selected else None for task in TASKS}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1 follows q-present-a/cells.py: task-wise reference/cost bars with labels.
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ratios = [refs[t] / picked_costs[t] if picked_costs[t] else 0 for t in TASKS]
    bars = ax.bar(TASKS, ratios, color=["#247ba0", "#70c1b3", "#f3a712"])
    ax.set(ylabel="Reference / selected receipt cost", title="Selected task cost ratios")
    ax.grid(axis="y", alpha=.15)
    for bar, value in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.3f}x", ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(figure_root / "selected-task-cost-ratios.png", dpi=130); plt.close(fig)

    # Figure 2 compares the exact target receipt and available PASS fallbacks.
    fig, ax = plt.subplots(figsize=(8, 4.2))
    xlabels, values, colors = [], [], []
    for task in TASKS:
        for index, receipt in enumerate(CONFIG["candidates"][task]):
            xlabels.append(task + (" target" if index == 0 else f" fallback {index}"))
            values.append(receipt["cost"])
            colors.append("#247ba0" if index == 0 else "#c96d48")
    bars = ax.bar(xlabels, values, color=colors)
    ax.set_yscale("log"); ax.set_ylabel("Recorded PASS cost (log scale)")
    ax.set_title("Receipt-bound target and fallback costs"); ax.tick_params(axis="x", rotation=28)
    ax.grid(axis="y", alpha=.15)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value, f"{value:.3g}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout(); fig.savefig(figure_root / "receipt-cost-comparison.png", dpi=130); plt.close(fig)

    # Figure 3 records how many exact members were selected task by task.
    fig, ax = plt.subplots(figsize=(8, 4.2))
    counts = []
    for task in TASKS:
        expected = sum(1 for name in CONFIG["required_members"] if name == task + ".kmb" or name.startswith(task + "/"))
        got = len(selected[task]["candidate"]["member_sha256"]) if task in selected else 0
        counts.append((got, expected))
    bars = ax.bar(TASKS, [x[0] for x in counts], color="#70c1b3")
    ax.set(ylabel="Hash-verified members", title="Complete task bundles selected")
    ax.grid(axis="y", alpha=.15)
    for bar, (got, expected) in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{got}/{expected}", ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(figure_root / "selected-member-counts.png", dpi=130); plt.close(fig)
    print("Figures:", *[str(p) for p in sorted(figure_root.glob("*.png"))], sep="\n  ")


def write_empty_archive(output_zip):
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in CONFIG["required_members"]:
            archive.writestr(name, b"")


def run_factory():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    paths = enumerate_inputs()
    archives = [p for p in paths if p.suffix.lower() == ".zip"]
    direct_groups = {}
    for path in paths:
        if path.suffix.lower() in (".kmb", ".kmx", ".npz"):
            direct_groups.setdefault(relative_source_group(path), []).append(path)
    selected = {}
    for task in TASKS:
        for candidate in CONFIG["candidates"][task]:
            references = find_candidate(candidate, archives, direct_groups)
            if references is not None:
                selected[task] = {"candidate": candidate, "references": references}
                break
    output_zip = OUTPUT_ROOT / "submission.zip"
    target_hashes = CONFIG["target_member_hashes"]
    fallback_count = 0
    chosen_hashes = {}
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in CONFIG["required_members"]:
            task = "invsqrt" if name == "invsqrt.kmb" else "dqi" if name == "dqi.kmb" else "dfthcsos"
            picked = selected.get(task)
            if picked is None:
                archive.writestr(name, b"")
                fallback_count += 1
                continue
            reference = picked["references"].get(name)
            if reference is None:
                archive.writestr(name, b"")
                fallback_count += 1
                continue
            copy_reference(archive, name, reference)
            chosen_hashes[name] = reference["sha256"]
            if reference["sha256"] != target_hashes[task].get(name):
                fallback_count += 1

    all_tasks_selected = all(task in selected for task in TASKS)
    metric = None
    if all_tasks_selected:
        metric = math.prod(CONFIG["reference_costs"][task] / selected[task]["candidate"]["cost"] for task in TASKS) ** (1 / 3)
        if not math.isfinite(metric):
            metric = None
    factory_check = {"metric": metric, "rows": 18, "fallbacks": fallback_count}
    (OUTPUT_ROOT / "factory_check.json").write_text(json.dumps(factory_check, sort_keys=True) + "\n", encoding="utf-8")

    report = {
        "status": "complete" if all_tasks_selected else "no_complete_receipt_set",
        "input_root": str(INPUT_ROOT),
        "output_zip": str(output_zip),
        "factory_check": factory_check,
        "selected_by_task": {
            task: {
                "label": selected[task]["candidate"]["label"],
                "cost": selected[task]["candidate"]["cost"],
                "status": selected[task]["candidate"]["status"],
                "verified_shots": selected[task]["candidate"]["verified_shots"],
                "receipt_path": selected[task]["candidate"]["receipt_path"],
                "source": selected[task]["candidate"]["source"],
                "member_sha256": selected[task]["candidate"]["member_sha256"],
                "input_files": {name: ref["path"] for name, ref in selected[task]["references"].items()},
            }
            for task in selected
        },
        "selected_member_sha256": chosen_hashes,
        "zip_sha256": hashlib.sha256(output_zip.read_bytes()).hexdigest(),
        "runtime_seconds": time.perf_counter() - STARTED,
        "metric_scope": "geometric mean of reference/receipt-cost for each task's exact selected hash bundle; null unless all three task bundles have a PASS receipt and matching bytes",
    }
    (OUTPUT_ROOT / "selection_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        write_figures(selected, metric)
    except BaseException as exc:
        print("Figure generation failed but submission outputs are complete:", type(exc).__name__, str(exc))
    with zipfile.ZipFile(output_zip) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        crc = archive.testzip()
    print("PICKS_JSON:", json.dumps(report["selected_by_task"], sort_keys=True))
    print("FACTORY_CHECK:", json.dumps(factory_check, sort_keys=True))
    print("ZIP_MEMBERS:", len(names), "EXPECTED:", len(CONFIG["required_members"]), "CRC_ERROR:", crc)
    print("ZIP_SHA256:", report["zip_sha256"])
    print("RUNTIME_SECONDS:", report["runtime_seconds"])


def fallback_after_exception(exc):
    print("Factory error caught at notebook top level:", type(exc).__name__, str(exc))
    try:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        output_zip = OUTPUT_ROOT / "submission.zip"
        write_empty_archive(output_zip)
        check = {"metric": None, "rows": 18, "fallbacks": 18}
        (OUTPUT_ROOT / "factory_check.json").write_text(json.dumps(check, sort_keys=True) + "\n", encoding="utf-8")
        report = {"status": "caught_exception_placeholder_archive", "factory_check": check, "error_type": type(exc).__name__, "error": str(exc), "zip_sha256": hashlib.sha256(output_zip.read_bytes()).hexdigest()}
        (OUTPUT_ROOT / "selection_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except BaseException as final_exc:
        print("Unable to write fallback artifacts:", type(final_exc).__name__, str(final_exc))


try:
    run_factory()
except BaseException as top_error:
    fallback_after_exception(top_error)
