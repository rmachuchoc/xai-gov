PY := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest
RUFF := .venv/bin/ruff
MYPY := .venv/bin/mypy

# Hermetic test runs. Pytest autoloads every pytest11 entry point it can
# see, and on this machine that includes ROS Jazzy's launch_testing (which
# then fails importing lark). A test result that depends on unrelated
# system packages is not reproducible, so autoload is off and the plugins
# this project actually uses are named explicitly.
# PYTHONPATH entries precede the venv's site-packages on sys.path, so a
# system package (ROS ships its own numpy) can shadow the pinned one. No
# project target needs anything from outside the venv, so every target runs
# with a cleared PYTHONPATH and user site-packages off.
SAFE_ENV := PYTHONPATH= PYTHONNOUSERSITE=1
PYTEST_ENV := $(SAFE_ENV) PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PYTEST_PLUGINS_ARGS :=
COV_PLUGIN := -p pytest_cov

.PHONY: install test cov lint types check info tree doctor doctor-raw run study study-amend study-quick sweeps figures m5 demand calibrate caltable clean

install:
	$(PIP) install --upgrade pip wheel setuptools
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

test:
	$(PYTEST_ENV) $(PYTEST) $(PYTEST_PLUGINS_ARGS)

cov:
	$(PYTEST_ENV) $(PYTEST) $(COV_PLUGIN) --cov=xai_gov --cov-report=term-missing

lint:
	$(SAFE_ENV) $(RUFF) check src tests scripts

types:
	$(SAFE_ENV) $(MYPY)

check: lint types test

info:
	$(SAFE_ENV) $(PY) -m xai_gov.cli.main info

# make run EXP=configs/experiments/pilot.yaml
EXP ?= configs/experiments/pilot.yaml
run:
	$(SAFE_ENV) $(PY) -m xai_gov.cli.main run --experiment $(EXP)

# The complete study: power check, sealed campaign, sensitivity, calibration.
# REPLICATES defaults to 40, not the protocol's 30: the power module computes
# 32 for a medium effect under a fixed-n paired test and 40 once the analysis
# is anytime-valid. Expect this to take a while — 7 arms x 40 replicates is
# 280 full runs.
REPLICATES ?= 40
WORKERS ?= 4
AMEND ?=
study:
	$(SAFE_ENV) $(PY) scripts/run_study.py --replicates $(REPLICATES) \
		--workers $(WORKERS) $(AMEND)

# Re-seal a changed plan as an amendment and run. Use this after editing the
# campaign: the superseded seal is kept beside the new one, so the record shows
# what was committed to first.
# Print-resolution figures for the manuscript. Needs matplotlib, which is not
# a project dependency: the campaign itself writes SVG and needs nothing.
DPI ?= 300
# The two parameter sweeps: break-even in the unit intervention cost, and the
# severity threshold. Separate from `study` because a sweep re-runs a paired
# campaign at every level, so it costs the study times the number of levels.
SWEEP_REPLICATES ?= 10
sweeps:
	$(SAFE_ENV) $(PY) scripts/run_sweeps.py --replicates $(SWEEP_REPLICATES)

# Turn a local copy of the M5 competition data into store x category demand
# series. The data is not in the repository: the competition terms permit use
# but not redistribution, so this target reads a directory you downloaded.
M5_SOURCE ?= $(HOME)/Downloads/m5
m5:
	$(SAFE_ENV) $(PY) scripts/fetch_m5.py --source $(M5_SOURCE) --level store_category

# Summarize the demand series before the cadence is sealed. The cadence
# determines the autocorrelation, so the decision has to be evidence-based and
# made before any calibration runs.
demand:
	$(SAFE_ENV) $(PY) scripts/inspect_demand.py
	@echo
	$(SAFE_ENV) $(PY) scripts/inspect_demand.py --weekly

# Calibrate against every real series and report the distribution of gaps.
# The cadence is explicit because it determines the autocorrelation being
# calibrated; the weekly default reflects the evidence in `make demand`, where
# the daily series show a cycle the twin has no term to represent.
CADENCE ?= weekly
PROCESS ?= both
calibrate:
	$(SAFE_ENV) $(PY) scripts/calibrate_all.py --cadence $(CADENCE) --process $(PROCESS)

# Build the calibration table and figure from whichever reports exist.
caltable:
	$(SAFE_ENV) $(PY) scripts/render_calibration.py --reports \
		outputs/calibration/calibration_weekly_both.json \
		outputs/calibration/calibration_weekly_ar1.json

figures:
	$(SAFE_ENV) $(PY) scripts/render_figures.py --dpi $(DPI)

study-amend:
	$(MAKE) study AMEND=--amend-seal

# A fast pass over the same pipeline, for checking the machinery before
# committing the compute. Under-powered on purpose, and the report says so.
study-quick:
	$(SAFE_ENV) $(PY) scripts/run_study.py \
		--campaign configs/campaigns/smoke.yaml \
		--replicates 3 --calibration-proposals 120 --allow-underpowered

tree:
	$(PY) -c "from xai_gov.core.paths import get_paths; get_paths().ensure_runtime_tree(); print('runtime tree ready')"

# Report anything in the environment that could make a run irreproducible.
doctor:
	@$(SAFE_ENV) $(PY) scripts/doctor.py

# The same audit with the raw shell environment, to see what leaks in.
doctor-raw:
	@$(PY) scripts/doctor.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
