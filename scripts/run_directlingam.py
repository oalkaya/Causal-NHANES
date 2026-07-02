from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from lingam import DirectLiNGAM
from lingam.utils import make_prior_knowledge

from utils.constraints import (
    get_forbidden_edges,
    get_required_edges,
    load_constraint_config,
)


# ---------------------------------------------------------------------
# Paths / settings
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "nhanes_clean.csv"
CONSTRAINT_DIR = PROJECT_ROOT / "constraints"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "directlingam"
GRAPH_CSV_DIR = OUTPUT_DIR / "graph_csvs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_CSV_DIR.mkdir(parents=True, exist_ok=True)


MEASURES = ["pwling"]
COEF_THRESHOLDS = [0.01, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
RANDOM_STATE = 42


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def safe_label(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def threshold_label(threshold: float) -> str:
    return f"{threshold:.3f}".replace(".", "p")


def load_constraint_configs(constraint_dir: Path) -> list[dict]:
    if not constraint_dir.exists():
        return []

    paths = sorted(constraint_dir.glob("*.yaml"))
    return [load_constraint_config(path) for path in paths]


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_processed_nhanes(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find processed NHANES file: {path}")

    df = pd.read_csv(path)
    variable_names = df.columns.tolist()

    X_raw = df.to_numpy(dtype=float)

    if np.isnan(X_raw).any():
        raise ValueError("Input contains NaNs. Run preprocessing first.")

    if np.isinf(X_raw).any():
        raise ValueError("Input contains infinite values. Run preprocessing first.")

    X_scaled = StandardScaler().fit_transform(X_raw)

    return df, X_scaled, variable_names


# ---------------------------------------------------------------------
# DirectLiNGAM-specific constraint conversion
# ---------------------------------------------------------------------

def make_directlingam_prior_knowledge(
    variable_names: list[str],
    forbidden_edges: list[tuple[str, str]],
):
    """
    Convert generic forbidden direct-edge directions into DirectLiNGAM prior knowledge.

    In lingam, no_paths uses index pairs:
        (source_index, target_index)

    meaning:
        source is not allowed to have a directed path to target.

    This is stronger than "no direct edge", but matches our tier logic:
    later-tier variables should not be ancestors of earlier-tier variables.
    """
    name_to_idx = {
        name: idx
        for idx, name in enumerate(variable_names)
    }

    no_paths = [
        (name_to_idx[source], name_to_idx[target])
        for source, target in forbidden_edges
    ]

    return make_prior_knowledge(
        n_variables=len(variable_names),
        no_paths=no_paths,
    )


# ---------------------------------------------------------------------
# DirectLiNGAM runner / graph conversion
# ---------------------------------------------------------------------

def run_directlingam(
    X: np.ndarray,
    measure: str,
    prior_knowledge=None,
) -> DirectLiNGAM:
    model = DirectLiNGAM(
        random_state=RANDOM_STATE,
        prior_knowledge=prior_knowledge,
        apply_prior_knowledge_softly=False,
        measure=measure,
    )

    model.fit(X)

    return model


def adjacency_to_edges(
    adjacency_matrix: np.ndarray,
    variable_names: list[str],
    coef_threshold: float,
) -> pd.DataFrame:
    """
    Convert DirectLiNGAM adjacency matrix into edge CSV format.

    lingam adjacency_matrix_[target, source] is the coefficient for:
        source -> target
    """
    rows = []

    n_vars = len(variable_names)

    for target_idx in range(n_vars):
        for source_idx in range(n_vars):
            if source_idx == target_idx:
                continue

            coef = adjacency_matrix[target_idx, source_idx]

            if abs(coef) < coef_threshold:
                continue

            rows.append({
                "source": variable_names[source_idx],
                "target": variable_names[target_idx],
                "edge_type": "directed",
                "weight": coef,
                "abs_weight": abs(coef),
            })

    return pd.DataFrame(
        rows,
        columns=[
            "source",
            "target",
            "edge_type",
            "weight",
            "abs_weight",
        ],
    )


# ---------------------------------------------------------------------
# Constraint checks
# ---------------------------------------------------------------------

def find_forbidden_edge_violations(
    edges: pd.DataFrame,
    forbidden_edges: list[tuple[str, str]],
) -> pd.DataFrame:
    """
    Check whether any output directed edge is in the forbidden edge list.

    This is only validation/reporting. It does not modify the graph.
    """
    if edges.empty:
        return pd.DataFrame(columns=edges.columns)

    forbidden = set(forbidden_edges)

    return edges[
        edges.apply(
            lambda row: (row["source"], row["target"]) in forbidden,
            axis=1,
        )
    ].copy()


def report_required_edges_if_present(
    required_edges: list[tuple[str, str]],
) -> None:
    """
    DirectLiNGAM prior knowledge cannot cleanly force direct edges.

    We intentionally do not post-hoc enforce required_edges.
    """
    if required_edges:
        print(
            "Required edges are defined in this constraint config, "
            "but DirectLiNGAM will not enforce them as direct edges."
        )


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------

def add_graph_column(edges: pd.DataFrame, graph_name: str) -> pd.DataFrame:
    edges = edges.copy()
    edges.insert(0, "graph", graph_name)
    return edges


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def print_edge_counts(graph_name: str, edges: pd.DataFrame) -> None:
    print()
    print("=" * 80)
    print(graph_name)
    print("=" * 80)

    if edges.empty:
        print("No edges.")
        return

    print(edges["edge_type"].value_counts().to_string())


def make_edge_summary(all_edges_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for graph_name, graph_edges in all_edges_df.groupby("graph", sort=True):
        directed = graph_edges[graph_edges["edge_type"] == "directed"]

        directed_edges = [
            f"{row.source} -> {row.target}"
            for row in directed.itertuples(index=False)
        ]

        rows.append({
            "graph": graph_name,
            "n_directed": len(directed_edges),
            "n_undirected": 0,
            "n_total": len(graph_edges),
            "directed_edges": "; ".join(directed_edges),
            "undirected_edges": "",
        })

    return pd.DataFrame(rows)


def make_edge_counts(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[
        [
            "graph",
            "n_directed",
            "n_undirected",
            "n_total",
        ]
    ].copy()


def save_causal_order(
    graph_name: str,
    model: DirectLiNGAM,
    variable_names: list[str],
) -> pd.DataFrame:
    order_rows = []

    for rank, idx in enumerate(model.causal_order_):
        order_rows.append({
            "graph": graph_name,
            "rank": rank,
            "variable": variable_names[idx],
            "variable_index": idx,
        })

    return pd.DataFrame(order_rows)


# ---------------------------------------------------------------------
# Run specifications
# ---------------------------------------------------------------------

def make_run_specs(variable_names: list[str]) -> list[dict]:
    """
    Build run specs:
      1. unconstrained baseline
      2. one native-constrained run for each YAML file in constraints/

    For DirectLiNGAM:
      - forbidden_edges become no_paths prior knowledge
      - required_edges are reported but not enforced as direct edges
    """
    specs = [
        {
            "constraint_name": "unconstrained",
            "prior_knowledge": None,
            "forbidden_edges": [],
            "required_edges": [],
        }
    ]

    for config in load_constraint_configs(CONSTRAINT_DIR):
        forbidden_edges = get_forbidden_edges(variable_names, config)
        required_edges = get_required_edges(variable_names, config)

        prior_knowledge = make_directlingam_prior_knowledge(
            variable_names=variable_names,
            forbidden_edges=forbidden_edges,
        )

        specs.append({
            "constraint_name": safe_label(config["name"]),
            "prior_knowledge": prior_knowledge,
            "forbidden_edges": forbidden_edges,
            "required_edges": required_edges,
        })

    return specs


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print(f"Loading processed NHANES from: {DATA_PATH}")

    df, X, variable_names = load_processed_nhanes(DATA_PATH)

    print("Data shape:", df.shape)
    print("Variables:", variable_names)

    run_specs = make_run_specs(variable_names)

    print()
    print("Constraint runs:")
    for spec in run_specs:
        print(f"- {spec['constraint_name']}")

    all_edges = []
    all_orders = []

    for measure in MEASURES:
        for threshold in COEF_THRESHOLDS:
            for spec in run_specs:
                graph_name = (
                    f"directlingam_"
                    f"measure_{safe_label(measure)}_"
                    f"threshold_{threshold_label(threshold)}_"
                    f"constraint_{spec['constraint_name']}"
                )

                report_required_edges_if_present(spec["required_edges"])

                model = run_directlingam(
                    X=X,
                    measure=measure,
                    prior_knowledge=spec["prior_knowledge"],
                )

                edges = adjacency_to_edges(
                    adjacency_matrix=model.adjacency_matrix_,
                    variable_names=variable_names,
                    coef_threshold=threshold,
                )

                print_edge_counts(graph_name, edges)

                if spec["constraint_name"] != "unconstrained":
                    violations = find_forbidden_edge_violations(
                        edges=edges,
                        forbidden_edges=spec["forbidden_edges"],
                    )

                    print(f"Forbidden-edge violations: {len(violations)}")

                    if not violations.empty:
                        print()
                        print("Forbidden-edge violations:")
                        print(
                            violations[
                                ["source", "target", "edge_type", "weight"]
                            ].to_string(index=False)
                        )

                graph_edges = add_graph_column(edges, graph_name)

                save_csv(
                    graph_edges,
                    GRAPH_CSV_DIR / f"{graph_name}_edges.csv",
                )

                all_edges.append(graph_edges)

                order_df = save_causal_order(
                    graph_name=graph_name,
                    model=model,
                    variable_names=variable_names,
                )

                all_orders.append(order_df)

    if not all_edges:
        raise RuntimeError("No DirectLiNGAM runs were completed.")

    all_edges_df = pd.concat(all_edges, ignore_index=True)
    all_orders_df = pd.concat(all_orders, ignore_index=True)

    save_csv(
        all_edges_df,
        OUTPUT_DIR / "directlingam_all_edges.csv",
    )

    save_csv(
        all_orders_df,
        OUTPUT_DIR / "directlingam_causal_orders.csv",
    )

    summary = make_edge_summary(all_edges_df)
    save_csv(
        summary,
        OUTPUT_DIR / "directlingam_edge_summary.csv",
    )

    counts = make_edge_counts(summary)
    save_csv(
        counts,
        OUTPUT_DIR / "directlingam_edge_counts.csv",
    )

    print()
    print("Compact edge summary:")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()