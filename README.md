# Causal-NHANES

Causal discovery and causal inference on NHANES data, with BMI as the outcome of interest.

The project compares three causal discovery methods—PC, DirectLiNGAM, and RESIT—then evaluates the resulting DAGs, identifies valid adjustment sets, and estimates average treatment effects on BMI with DoWhy.


# Setup

Create the Conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate causalinference
```

Run commands from the repository root using Python module execution:

```bash
python -m scripts.<script_name>
python -m utils.<utility_name>
```


# Project structure

The repository separates causal discovery and inference scripts from reusable utilities, background-knowledge constraints, datasets, and generated outputs.

```text
Causal-NHANES/
├── constraints/
│   ├── moderate_constraints.yaml
│   └── strict_constraints.yaml
├── data/
│   ├── nhanes.csv
│   └── processed/
├── outputs/
│   ├── adjustment_sets/
│   ├── ate/
│   ├── directlingam/
│   ├── pc/
│   └── resit/
├── scripts/
│   ├── run_directlingam.py
│   ├── run_inference.py
│   ├── run_pc.py
│   └── run_resit.py
├── utils/
│   ├── check_treatment_paths.py
│   ├── constraints.py
│   ├── find_adjustment_sets.py
│   ├── preprocess_nhanes.py
│   ├── test_adjustment_sets.py
│   └── visualize_graph_csv.py
├── deliverables/
...
```

- `constraints/` contains the moderate and strict background-knowledge configurations used during causal discovery.
- `data/` contains the raw NHANES dataset and the cleaned dataset produced by preprocessing.
- `scripts/` contains the main causal discovery and ATE estimation pipelines.
- `utils/` contains preprocessing, validation, graph inspection, visualization, and adjustment-set utilities.
- `outputs/` contains generated graphs, summaries, adjustment sets, visualizations, and inference results.
- `deliverables/` contains the project report and presentation.


# Workflow

1. **NHANES dataset preprocessing**  
   `utils/preprocess_nhanes.py`

2. **Causal discovery with PC, DirectLiNGAM, and RESIT**  
   `scripts/run_pc.py`, `scripts/run_directlingam.py`, `scripts/run_resit.py`  
   Background-knowledge constraints are handled by `utils/constraints.py`.

3. **Graph inspection and DAG selection**  
   `utils/check_treatment_paths.py`, `utils/visualize_graph_csv.py`

4. **Treatment-specific adjustment-set identification**  
   `utils/find_adjustment_sets.py`

5. **ATE estimation on BMI with DoWhy**  
   `scripts/run_inference.py`

# Scripts

### Causal discovery

The discovery scripts read `data/processed/nhanes_clean.csv`, run unconstrained and constraint-aware experiments, and save graph edge lists and summary tables under their respective output folders.

#### `scripts/run_pc.py`

Runs the constraint-based PC algorithm over the configured significance levels.


```bash
python -m scripts.run_pc
```

Outputs are written under `outputs/pc/`.

---

#### `scripts/run_directlingam.py`

Runs DirectLiNGAM using the pairwise likelihood measure and multiple coefficient thresholds.


```bash
python -m scripts.run_directlingam
```

Outputs are written under `outputs/directlingam/`.

The selected graph used for inference is:

```text
outputs/directlingam/graph_csvs/directlingam_measure_pwling_threshold_0p070_constraint_nhanes_strict_tiers_edges.csv
```

---

#### `scripts/run_resit.py`

Runs the nonlinear RESIT causal discovery method over the configured independence-test significance levels.


```bash
python -m scripts.run_resit
```

Outputs are written under `outputs/resit/`.

### Causal inference

#### `scripts/run_inference.py`

Reads the processed data and selected DAG, identifies or validates backdoor adjustment sets, and estimates ATEs on BMI using DoWhy linear regression.

The treatment contrasts are:

- `Vigorous_Activity`: binary contrast from 0 to 1
- `Total_Calories`: effect per +100 kcal
- `Protein_g`: effect per +10 g
- `Carbohydrates_g`: effect per +10 g

**With auto-selected adjustment sets**

```bash
python -m scripts.run_inference
```

**With manually selected adjustment sets**

```bash
python -m scripts.run_inference \
  --manual-adjustment "Carbohydrates_g=Gender,Total_Sugars_g" \
  --manual-adjustment "Protein_g=Carbohydrates_g,Gender,Total_Sugars_g" \
  --manual-adjustment "Total_Calories=Dietary_Fiber_g,Income_Ratio,Vigorous_Activity" \
  --manual-adjustment "Vigorous_Activity=Age,Gender,Income_Ratio"
```

Manual sets are validated against the selected DAG before estimation. A set is rejected if it contains unknown variables, treatment descendants, the treatment or outcome, or fails to block all backdoor paths.

Add `--run-refuters` to run the DoWhy refutation checks.

Outputs include:

```text
ate_results.csv
ate_results_compact.csv
ate_adjustment_sets.csv
ate_estimands.txt
ate_refuters.csv
```

`ate_refuters.csv` is only produced when `--run-refuters` is supplied.

# Utilities

#### `utils/preprocess_nhanes.py`

Reads the raw NHANES dataset, standardizes variable names and numeric representations, removes invalid or incomplete rows, and writes the cleaned dataset to `data/processed/nhanes_clean.csv`.


```bash
python -m utils.preprocess_nhanes
```

---

#### `utils/check_treatment_paths.py`

Reads every graph CSV in a directory and checks whether each predefined treatment has a directed path to BMI.


```bash
python -m utils.check_treatment_paths \
  outputs/directlingam/graph_csvs
```

The path-check summary is saved inside the supplied method output directory.

---

#### `utils/visualize_graph_csv.py`

Reads graph edge CSVs from a directory and renders each graph as an image.


```bash
python -m utils.visualize_graph_csv \
  outputs/directlingam/graph_csvs \
  outputs/directlingam/graph_images
```

---

#### `utils/find_adjustment_sets.py`

Takes a DAG edge CSV, treatment, and outcome. It constructs the treatment-specific backdoor graph, tests candidate sets using d-separation, and outputs all inclusion-minimal valid adjustment sets.


```bash
python -m utils.find_adjustment_sets \
  outputs/directlingam/graph_csvs/directlingam_measure_pwling_threshold_0p070_constraint_nhanes_strict_tiers_edges.csv \
  Vigorous_Activity \
  BMI
```

Results are written under `outputs/adjustment_sets/`.

Run the same command with `Total_Calories`, `Protein_g`, or `Carbohydrates_g` to obtain adjustment sets for the other treatment–outcome pairs.

---

#### `utils/constraints.py`

Loads and validates the YAML background-knowledge configurations and converts them into allowed, forbidden, and required edge definitions used by the discovery scripts.

---

#### `utils/test_adjustment_sets.py`

Tests the adjustment-set implementation on small DAGs with known expected solutions.


```bash
pytest -q utils/test_adjustment_sets.py
```
