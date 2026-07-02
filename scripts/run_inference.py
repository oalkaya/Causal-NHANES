from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from dowhy import CausalModel


# Supports both:
#   python scripts/run_inference.py
#   python -m scripts.run_inference
if __package__:
    from scripts.find_adjustment_sets import (
        is_valid_adjustment_set,
        make_backdoor_graph,
    )
else:
    from find_adjustment_sets import (
        is_valid_adjustment_set,
        make_backdoor_graph,
    )


warnings.filterwarnings(
    "ignore",
    message="DataFrameGroupBy.apply operated on the grouping columns.*",
    category=FutureWarning,
    module="dowhy.causal_estimator",
)


# ---------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "nhanes_clean.csv"

DEFAULT_GRAPH_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "directlingam"
    / "graph_csvs"
    / (
        "directlingam_measure_pwling_threshold_0p070_"
        "constraint_nhanes_strict_tiers_edges.csv"
    )
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ate"

OUTCOME = "BMI"

TREATMENTS = {
    "Vigorous_Activity": {
        "label": "Treatment A",
        "type": "binary",
        "report_unit": "0 -> 1",
        "control_value": 0,
        "treatment_value": 1,
        "scale": 1.0,
        "scaled_name": "Vigorous_Activity",
    },
    "Total_Calories": {
        "label": "Treatment B",
        "type": "continuous",
        "report_unit": "+100 kcal",
        "control_value": 0,
        "treatment_value": 1,
        "scale": 100.0,
        "scaled_name": "Total_Calories_per_100kcal",
    },
    "Protein_g": {
        "label": "Treatment C",
        "type": "continuous",
        "report_unit": "+10 g",
        "control_value": 0,
        "treatment_value": 1,
        "scale": 10.0,
        "scaled_name": "Protein_per_10g",
    },
    "Carbohydrates_g": {
        "label": "Treatment D",
        "type": "continuous",
        "report_unit": "+10 g",
        "control_value": 0,
        "treatment_value": 1,
        "scale": 10.0,
        "scaled_name": "Carbohydrates_per_10g",
    },
}


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def safe_label(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def graph_name_from_path(path: Path) -> str:
    return path.stem.removesuffix("_edges")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    if df.isna().any().any():
        raise ValueError("Input data contains NaNs. Run preprocessing first.")

    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if non_numeric:
        raise ValueError(
            f"Input data contains non-numeric columns: {non_numeric}"
        )

    if np.isinf(df.to_numpy(dtype=float)).any():
        raise ValueError(
            "Input data contains infinite values. Run preprocessing first."
        )

    return df


def load_edge_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Graph CSV not found: {path}")

    edges = pd.read_csv(path)

    required_columns = {"source", "target", "edge_type"}
    missing = required_columns - set(edges.columns)

    if missing:
        raise ValueError(
            f"Graph CSV is missing required columns: {sorted(missing)}"
        )

    return edges


def rename_treatment_in_edges(
    edges: pd.DataFrame,
    original_treatment: str,
    scaled_treatment: str,
) -> pd.DataFrame:
    edges = edges.copy()

    replacement = {
        original_treatment: scaled_treatment,
    }

    edges["source"] = edges["source"].replace(replacement)
    edges["target"] = edges["target"].replace(replacement)

    return edges


def make_model_data(
    df: pd.DataFrame,
    original_treatment: str,
    treatment_spec: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    """
    Keep the binary treatment as 0/1.

    Rescale continuous treatments so that a one-unit treatment contrast means:
      - 100 kcal for Total_Calories
      - 10 g for Protein_g
      - 10 g for Carbohydrates_g
    """
    df_model = df.copy()

    scaled_treatment = treatment_spec["scaled_name"]
    scale = treatment_spec["scale"]

    if original_treatment != scaled_treatment:
        df_model[scaled_treatment] = (
            df_model[original_treatment] / scale
        )
        df_model = df_model.drop(columns=[original_treatment])

    return df_model, scaled_treatment


# ---------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------

def build_digraph(
    edges: pd.DataFrame,
    all_nodes: list[str],
) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(all_nodes)

    directed_edges = edges[edges["edge_type"] == "directed"]

    graph.add_edges_from(
        zip(
            directed_edges["source"],
            directed_edges["target"],
        )
    )

    return graph


def validate_graph_for_ate(
    graph: nx.DiGraph,
    treatment: str,
    outcome: str,
) -> dict[str, Any]:
    is_dag = nx.is_directed_acyclic_graph(graph)

    has_directed_path = (
        treatment in graph
        and outcome in graph
        and nx.has_path(graph, treatment, outcome)
    )

    directed_path = (
        nx.shortest_path(graph, treatment, outcome)
        if has_directed_path
        else []
    )

    return {
        "is_dag": is_dag,
        "has_directed_path_to_outcome": has_directed_path,
        "directed_path_to_outcome": " -> ".join(directed_path),
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
    }


def graph_to_gml_string(graph: nx.DiGraph) -> str:
    nodes = list(graph.nodes())
    node_to_id = {
        node: index
        for index, node in enumerate(nodes)
    }

    lines = [
        "graph [",
        "  directed 1",
    ]

    for node in nodes:
        lines.extend([
            "  node [",
            f"    id {node_to_id[node]}",
            f'    label "{node}"',
            "  ]",
        ])

    for source, target in graph.edges():
        lines.extend([
            "  edge [",
            f"    source {node_to_id[source]}",
            f"    target {node_to_id[target]}",
            "  ]",
        ])

    lines.append("]")

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Manual adjustment-set handling
# ---------------------------------------------------------------------

def parse_manual_adjustments(
    values: list[str],
) -> dict[str, list[str]]:
    """
    Parse repeated arguments in the form:

        Treatment=Variable1,Variable2

    Examples:

        Vigorous_Activity=Age,Gender,Sedentary_Minutes
        Total_Calories=Vigorous_Activity,Dietary_Fiber_g
    """
    manual_adjustments: dict[str, list[str]] = {}

    for value in values:
        treatment, separator, variables_text = value.partition("=")

        treatment = treatment.strip()

        if not separator or not treatment:
            raise ValueError(
                "Manual adjustment sets must use the format "
                "'Treatment=Variable1,Variable2'."
            )

        if treatment in manual_adjustments:
            raise ValueError(
                f"A manual adjustment set was specified more than once "
                f"for {treatment!r}."
            )

        variables = [
            variable.strip()
            for variable in variables_text.split(",")
            if variable.strip()
        ]

        if len(variables) != len(set(variables)):
            raise ValueError(
                f"Manual adjustment set for {treatment!r} "
                "contains duplicate variables."
            )

        manual_adjustments[treatment] = variables

    return manual_adjustments


def validate_manual_adjustment_set(
    graph: nx.DiGraph,
    treatment: str,
    outcome: str,
    adjustment_set: list[str],
) -> None:
    """
    Validate a manually supplied adjustment set under the selected DAG.

    The set must:
      - contain known graph variables,
      - exclude treatment and outcome,
      - exclude descendants of the treatment,
      - block all backdoor paths.
    """
    adjustment_variables = set(adjustment_set)

    unknown_variables = adjustment_variables - set(graph.nodes)

    if unknown_variables:
        raise ValueError(
            "Manual adjustment set contains variables not present "
            f"in the graph: {sorted(unknown_variables)}"
        )

    invalid_endpoints = adjustment_variables & {
        treatment,
        outcome,
    }

    if invalid_endpoints:
        raise ValueError(
            "Manual adjustment set cannot contain the treatment "
            f"or outcome: {sorted(invalid_endpoints)}"
        )

    descendants = nx.descendants(graph, treatment)
    descendant_adjustments = adjustment_variables & descendants

    if descendant_adjustments:
        raise ValueError(
            "Manual adjustment set contains descendants of the treatment: "
            f"{sorted(descendant_adjustments)}. These may be mediators "
            "or other post-treatment variables."
        )

    backdoor_graph = make_backdoor_graph(
        graph=graph,
        treatment=treatment,
    )

    is_valid = is_valid_adjustment_set(
        backdoor_graph=backdoor_graph,
        treatment=treatment,
        outcome=outcome,
        adjustment_set=adjustment_variables,
    )

    if not is_valid:
        raise ValueError(
            f"Manual adjustment set {sorted(adjustment_variables)} "
            f"is not valid for {treatment} -> {outcome}. "
            "It does not block all backdoor paths in the selected DAG."
        )


# ---------------------------------------------------------------------
# DoWhy helpers
# ---------------------------------------------------------------------

def extract_backdoor_variables(
    identified_estimand,
) -> list[str]:
    """
    DoWhy versions differ slightly in how they expose adjustment variables.
    """
    try:
        variables = identified_estimand.get_backdoor_variables()

        if variables is not None:
            return list(variables)
    except Exception:
        pass

    try:
        variables = identified_estimand.backdoor_variables

        if variables is not None:
            return list(variables)
    except Exception:
        pass

    try:
        backdoor_info = identified_estimand.estimands.get(
            "backdoor",
            {},
        )

        variables = backdoor_info.get(
            "backdoor_variables",
            [],
        )

        if variables is not None:
            return list(variables)
    except Exception:
        pass

    return []


def safe_get_confidence_interval(
    estimate,
) -> tuple[float | None, float | None]:
    try:
        ci = estimate.get_confidence_intervals()

        if ci is None:
            return None, None

        values = np.asarray(ci).flatten()

        if len(values) >= 2:
            return float(values[0]), float(values[1])

    except Exception:
        pass

    return None, None


def safe_get_p_value(
    estimate,
) -> float | None:
    try:
        significance = estimate.test_stat_significance()

        if not isinstance(significance, dict):
            return None

        p_value = significance.get("p_value")

        if p_value is None:
            return None

        values = np.asarray(p_value).flatten()

        if len(values) >= 1:
            return float(values[0])

    except Exception:
        pass

    return None


# ---------------------------------------------------------------------
# ATE estimation
# ---------------------------------------------------------------------

def estimate_ate_for_treatment(
    df: pd.DataFrame,
    base_edges: pd.DataFrame,
    graph_csv: Path,
    original_treatment: str,
    treatment_spec: dict[str, Any],
    run_refuters: bool,
    manual_adjustment_set: list[str] | None,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    str,
]:
    df_model, treatment = make_model_data(
        df=df,
        original_treatment=original_treatment,
        treatment_spec=treatment_spec,
    )

    edges = rename_treatment_in_edges(
        edges=base_edges,
        original_treatment=original_treatment,
        scaled_treatment=treatment,
    )

    graph = build_digraph(
        edges=edges,
        all_nodes=df_model.columns.tolist(),
    )

    graph_checks = validate_graph_for_ate(
        graph=graph,
        treatment=treatment,
        outcome=OUTCOME,
    )

    if not graph_checks["is_dag"]:
        raise ValueError(
            f"Selected graph is not a DAG for "
            f"{original_treatment} -> {OUTCOME}."
        )

    if not graph_checks["has_directed_path_to_outcome"]:
        print(
            f"Warning: no directed path from {treatment} to {OUTCOME}. "
            "The selected DAG implies no directed causal route."
        )

    if manual_adjustment_set is None:
        adjustment_mode = "graph"

        model = CausalModel(
            data=df_model,
            treatment=treatment,
            outcome=OUTCOME,
            graph=graph_to_gml_string(graph),
            effect_modifiers=[],
        )

        identified_estimand = model.identify_effect(
            proceed_when_unidentifiable=False,
        )

        adjustment_set = extract_backdoor_variables(
            identified_estimand
        )

    else:
        adjustment_mode = "manual"

        validate_manual_adjustment_set(
            graph=graph,
            treatment=treatment,
            outcome=OUTCOME,
            adjustment_set=manual_adjustment_set,
        )

        missing_from_data = sorted(
            set(manual_adjustment_set)
            - set(df_model.columns)
        )

        if missing_from_data:
            raise ValueError(
                "Manual adjustment variables are missing from the data: "
                f"{missing_from_data}"
            )

        adjustment_set = list(manual_adjustment_set)

        print(
            "Validated manual adjustment set: "
            f"{adjustment_set}"
        )

        model = CausalModel(
            data=df_model,
            treatment=treatment,
            outcome=OUTCOME,
            common_causes=adjustment_set,
            effect_modifiers=[],
        )

        identified_estimand = model.identify_effect(
            proceed_when_unidentifiable=False,
        )

    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        target_units="ate",
        control_value=treatment_spec["control_value"],
        treatment_value=treatment_spec["treatment_value"],
        effect_modifiers=[],
        confidence_intervals=True,
        test_significance=True,
    )

    ci_low, ci_high = safe_get_confidence_interval(estimate)
    p_value = safe_get_p_value(estimate)

    result = {
        "treatment_label": treatment_spec["label"],
        "original_treatment": original_treatment,
        "model_treatment": treatment,
        "treatment_type": treatment_spec["type"],
        "outcome": OUTCOME,
        "report_unit": treatment_spec["report_unit"],
        "ate": float(estimate.value),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "estimator": "backdoor.linear_regression",
        "adjustment_mode": adjustment_mode,
        "manual_adjustment_validated": (
            adjustment_mode == "manual"
        ),
        "adjustment_set": "; ".join(adjustment_set),
        "n_adjustment_variables": len(adjustment_set),
        "graph_name": graph_name_from_path(graph_csv),
        "graph_csv": str(graph_csv),
        **graph_checks,
    }

    refuter_rows: list[dict[str, Any]] = []

    if run_refuters:
        refuter_names = [
            "random_common_cause",
            "placebo_treatment_refuter",
            "data_subset_refuter",
        ]

        for refuter_name in refuter_names:
            print(
                f"[{original_treatment}] "
                f"Running refuter: {refuter_name}"
            )

            try:
                refutation = model.refute_estimate(
                    identified_estimand,
                    estimate,
                    method_name=refuter_name,
                )

                summary = str(refutation)

            except Exception as exc:
                summary = f"FAILED: {exc}"

            refuter_rows.append({
                "original_treatment": original_treatment,
                "model_treatment": treatment,
                "outcome": OUTCOME,
                "adjustment_mode": adjustment_mode,
                "refuter": refuter_name,
                "refutation_summary": summary,
            })

    estimand_text = (
        "\n"
        + "=" * 80
        + f"\nTreatment: {original_treatment}\n"
        + f"Adjustment mode: {adjustment_mode}\n"
        + f"Adjustment set: {adjustment_set}\n"
        + "=" * 80
        + "\n\n"
        + str(identified_estimand)
        + "\n\nEstimate:\n"
        + str(estimate)
        + "\n"
    )

    return result, refuter_rows, estimand_text


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

def run_ate_pipeline(
    data_path: Path,
    graph_csv: Path,
    output_dir: Path,
    run_refuters: bool,
    manual_adjustments: dict[str, list[str]],
) -> None:
    df = load_data(data_path)
    base_edges = load_edge_csv(graph_csv)

    graph_name = graph_name_from_path(graph_csv)
    output_name = safe_label(graph_name)

    if manual_adjustments:
        output_name += "_manual"

    graph_output_dir = output_dir / output_name
    graph_output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    refuter_rows: list[dict[str, Any]] = []
    estimand_texts: list[str] = []

    print(f"Data:   {data_path}")
    print(f"Graph:  {graph_csv}")
    print(f"Output: {graph_output_dir}")

    for treatment, treatment_spec in TREATMENTS.items():
        print()
        print("=" * 80)
        print(
            f"{treatment_spec['label']}: "
            f"{treatment} -> {OUTCOME}"
        )
        print("=" * 80)

        result, treatment_refuters, estimand_text = (
            estimate_ate_for_treatment(
                df=df,
                base_edges=base_edges,
                graph_csv=graph_csv,
                original_treatment=treatment,
                treatment_spec=treatment_spec,
                run_refuters=run_refuters,
                manual_adjustment_set=manual_adjustments.get(
                    treatment
                ),
            )
        )

        results.append(result)
        refuter_rows.extend(treatment_refuters)
        estimand_texts.append(estimand_text)

        print(
            f"Adjustment mode: {result['adjustment_mode']}"
        )
        print(
            f"Adjustment set:  {result['adjustment_set']}"
        )
        print(
            f"ATE ({result['report_unit']}): {result['ate']}"
        )
        print(
            f"Directed path:   "
            f"{result['directed_path_to_outcome']}"
        )

    results_df = pd.DataFrame(results)

    results_path = graph_output_dir / "ate_results.csv"
    results_df.to_csv(results_path, index=False)

    compact_columns = [
        "treatment_label",
        "original_treatment",
        "treatment_type",
        "report_unit",
        "outcome",
        "ate",
        "ci_low",
        "ci_high",
        "p_value",
        "adjustment_mode",
        "manual_adjustment_validated",
        "n_adjustment_variables",
        "adjustment_set",
        "has_directed_path_to_outcome",
        "directed_path_to_outcome",
    ]

    compact_results = results_df[compact_columns].copy()

    compact_path = graph_output_dir / "ate_results_compact.csv"
    compact_results.to_csv(compact_path, index=False)

    adjustment_sets = results_df[
        [
            "original_treatment",
            "model_treatment",
            "outcome",
            "adjustment_mode",
            "manual_adjustment_validated",
            "adjustment_set",
            "n_adjustment_variables",
        ]
    ].copy()

    adjustment_sets_path = (
        graph_output_dir
        / "ate_adjustment_sets.csv"
    )

    adjustment_sets.to_csv(
        adjustment_sets_path,
        index=False,
    )

    estimands_path = graph_output_dir / "ate_estimands.txt"

    estimands_path.write_text(
        "\n".join(estimand_texts),
        encoding="utf-8",
    )

    if run_refuters:
        refuters_path = graph_output_dir / "ate_refuters.csv"

        pd.DataFrame(refuter_rows).to_csv(
            refuters_path,
            index=False,
        )

        print(f"Saved refuters:        {refuters_path}")

    print()
    print("=" * 80)
    print("ATE summary")
    print("=" * 80)
    print(compact_results.to_string(index=False))

    print()
    print(f"Saved full results:    {results_path}")
    print(f"Saved compact results: {compact_path}")
    print(f"Saved adjustment sets: {adjustment_sets_path}")
    print(f"Saved estimands:       {estimands_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate graph-based or manually adjusted ATEs "
            "for the predefined NHANES treatments."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="Path to the cleaned NHANES CSV.",
    )

    parser.add_argument(
        "--graph-csv",
        type=Path,
        default=DEFAULT_GRAPH_CSV,
        help="Path to the selected DAG edge CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Root output directory.",
    )

    parser.add_argument(
        "--run-refuters",
        action="store_true",
        help="Run DoWhy refutation checks.",
    )

    parser.add_argument(
        "--manual-adjustment",
        action="append",
        default=[],
        metavar="TREATMENT=VARIABLE1,VARIABLE2",
        help=(
            "Use a manual adjustment set for one treatment. "
            "The argument may be repeated. Example: "
            "--manual-adjustment "
            "'Vigorous_Activity=Age,Gender,Sedentary_Minutes'"
        ),
    )

    args = parser.parse_args()

    manual_adjustments = parse_manual_adjustments(
        args.manual_adjustment
    )

    unknown_treatments = (
        set(manual_adjustments)
        - set(TREATMENTS)
    )

    if unknown_treatments:
        raise ValueError(
            "Manual adjustment sets were supplied for unknown "
            f"treatments: {sorted(unknown_treatments)}"
        )

    run_ate_pipeline(
        data_path=args.data,
        graph_csv=args.graph_csv,
        output_dir=args.output_dir,
        run_refuters=args.run_refuters,
        manual_adjustments=manual_adjustments,
    )


if __name__ == "__main__":
    main()