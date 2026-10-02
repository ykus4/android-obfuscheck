# android-obfuscheck

A CI gate for Android release builds: it measures how much of your APK/AAB R8/ProGuard actually
obfuscated, and can confirm with [APKiD](https://github.com/rednaga/APKiD) which toolchain built it.

A missing `minifyEnabled true`, a `-keep class com.example.** { *; }` that is too broad, or a
library whose consumer rules keep everything all fail silently: the build still succeeds. android-obfuscheck
turns them into a failing check with a per-package breakdown.

Inspired by [liansecurityOS/apk-obfucation-detection](https://github.com/liansecurityOS/apk-obfucation-detection),
which classifies each class as obfuscated or not from its class, method and field names and
reports the share of obfuscated classes.

```
$ android-obfuscheck scan app-release.apk --min-coverage 0.8 --require-compiler r8
target:     app-release.apk
detector:   ml
classes:    5280 total, 4920 in scope
coverage:   89.3% (4393/4920)

package                                           classes  coverage
org.joda.time                                         248      0.0%
T                                                     157     99.4%
com.ezylang.evalex                                    148      4.7%
A                                                     137     99.3%
...

apkid 3.1.0:
  anti_vm          Build.FINGERPRINT check, Build.MANUFACTURER check
  compiler         r8
  manipulator      Resources Confusion

PASS
```

## Install

```sh
pip install 'android-obfuscheck[ml] @ git+https://github.com/ykus4/android-obfuscheck'
pip install apkid    # optional, for --apkid / --require-compiler / --forbid
```

## Usage

```sh
android-obfuscheck scan app-release.apk                         # report only
android-obfuscheck scan app.aab --min-coverage 0.7              # exit 1 below 70 %
android-obfuscheck scan app.apk --baseline main.json --max-drop 0.03
android-obfuscheck scan app.apk --include com.example --list-unobfuscated 50
android-obfuscheck scan app.apk --require-compiler r8 --forbid packer
android-obfuscheck scan app.apk --format markdown >> "$GITHUB_STEP_SUMMARY"
```

| option | meaning |
|---|---|
| `--min-coverage F` | fail if the obfuscated share of in-scope classes is below `F` (0–1) |
| `--baseline FILE` / `--max-drop F` | compare with a previous `--json-out` report; fail on a drop larger than `F` |
| `--include P` / `--exclude P` | package prefixes to count or skip (repeatable, segment-aware) |
| `--no-default-excludes` | also count `androidx.`, `kotlin.`, `com.google.`, … (skipped by default) |
| `--list-unobfuscated N` | list up to N in-scope classes that stayed readable, to audit keep rules |
| `--model FILE` | use a trained ML model instead of the built-in heuristic |
| `--apkid` | add APKiD's compiler/obfuscator/packer/anti-analysis findings to the report |
| `--require-compiler NAME` | fail unless APKiD reports e.g. `r8` (repeatable, any match passes) |
| `--forbid CATEGORY` | fail if APKiD reports e.g. `packer`, `protector`, `anti_debug` |
| `--json-out` / `--markdown-out` | write reports (Markdown is appended, for `$GITHUB_STEP_SUMMARY`) |

Exit codes: `0` pass, `1` a gate failed, `2` input or tool error.

Accepts `.apk`, `.aab`, any zip with `.dex` entries, or a raw `.dex`. Every dex is read, including
multidex and AAB feature modules.

### Choosing the scope

R8 often repackages obfuscated classes into short top-level packages (`A`, `C5`, `T` above), so
`--include com.example` misses them. The default (all classes except well-known libraries) is
usually the better scope; add `--exclude` for libraries you keep on purpose.

### APKiD

APKiD fingerprints the dex with YARA rules, which complements name-based coverage:

- `compiler: r8` means the release build really went through R8. `dx`/`d8` means minification was
  off. `dexlib` means the APK was rebuilt with apktool/smali, i.e. it is not what your build produced.
- `obfuscator`/`protector`/`packer` show commercial tools such as DexGuard.
- `anti_vm`/`anti_debug` usually come from SDKs, so they are informational.

`manipulator: Resources Confusion` also matches ordinary AGP builds with resource path shortening,
so check your own release build before adding `--forbid manipulator`.

APKiD is GPL-3.0 and is run as a separate `apkid` executable, not imported.

## Detectors

| detector | needs | accuracy* |
|---|---|---:|
| `heuristic` (default) | nothing | 98.2 % |
| `ml` (`--model`, trained with `--algorithm lr`) | `android-obfuscheck[ml]` and a trained model | 99.3 % |
| `ml` (`--algorithm lightgbm`) | `android-obfuscheck[lightgbm]` and a trained model | 99.3 % |

\* Grouped, stratified 5-fold CV on the upstream `new_train.csv` (14,619 labelled classes).
[docs/model-experiments.md](docs/model-experiments.md) compares 18 detectors, including the
upstream LSTM, and explains the choices below.

The heuristic flags a class when its name looks like an R8/ProGuard name (`a`, `Zb`, `b12`, `A0`,
`zza`, …), when an inner class has such a name, or when at least 40 % of its members do. It skips
`R`/`BuildConfig` and synthetic helpers, and ignores framework-mandated members such as `run` and
`toString`. The ML model combines character 1–4-gram TF-IDF with 25 shape features. A package
rule backs it up: classes whose package and name both look renamed (`d4.d`) always count as
obfuscated, which matters for member-less interfaces the training data rarely covers.

### Training a model

```sh
scripts/fetch_dataset.sh data/new_train.csv
android-obfuscheck train --data data/new_train.csv --out android-obfuscheck-model.joblib
# or: --algorithm lightgbm   (pip install 'android-obfuscheck[lightgbm]')
android-obfuscheck scan app.apk --model android-obfuscheck-model.joblib
```

The upstream repository has no license, so its dataset is downloaded on demand rather than
vendored. Check its terms before redistributing a model trained on it. Models are joblib pickles,
so only load ones you trust. To train on your own data, use a CSV with `text,label` columns, where
`text` is `Class: <name>; Method: <m1> <m2> Field: <f1>`.

## GitHub Action

```yaml
- uses: ykus4/android-obfuscheck@v0.1.0
  with:
    path: app/build/outputs/apk/release/app-release-unsigned.apk
    min-coverage: "0.6"
    require-compiler: r8
    forbid: packer
```

The step writes `android-obfuscheck-report.json`, appends a summary to the job page, and sets the
`coverage` and `passed` outputs. See [`examples/android-release.yml`](examples/android-release.yml)
for a workflow that keeps the latest `main` report as a baseline for pull requests.

## Development

```sh
uv venv && uv pip install -e '.[dev]'
.venv/bin/pytest -q
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

```
src/android_obfuscheck/
  classinfo.py   ClassInfo record, parser for the upstream CSV format
  extract.py     dex extraction from apk/aab/zip/dex via androguard
  detectors.py   HeuristicDetector, MLDetector, sklearn pipeline
  train.py       dataset loading, hold-out evaluation, model export
  apkid.py       APKiD subprocess runner, result parsing, gates
  report.py      scoping, per-package aggregation, gates, text/Markdown/JSON rendering
  cli.py         `android-obfuscheck scan` / `android-obfuscheck train`
experiments/     model comparison scripts and results (docs/model-experiments.md)
action.yml       composite GitHub Action
```
