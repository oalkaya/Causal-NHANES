from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

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

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pc"
GRAPH_CSV_DIR = OUTPUT_DIR / "graph_csvs"

ALPHAS = [0.001, 0.01, 0.05, 0.6, 0.7, 0.8, 0.9, 0.10]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_CSV_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def safe_label(text: str) -> str:
    """
    Make text safe for filenames.
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def alpha_label(alpha: float) -> str:
    """
    Fixed-width alpha label for filename sorting.

    0.001 -> 0p001
    0.010 -> 0p010
    0.050 -> 0p050
    """
    return f"{alpha:.3f}".replace(".", "p")


def load_constraint_configs(constraint_dir: Path) -> list[dict]:
    """
    Load every YAML constraint config in constraints/.
    If the folder is missing or empty, return an empty list.
    """
    if not constraint_dir.exists():
        return []

    paths = sorted(constraint_dir.glob("*.yaml"))

    return [load_constraint_config(path) for path in paths]


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_processed_nhanes(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    Load already-cleaned NHANES data and standardize it.
    """
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
# PC-specific constraint conversion
# ---------------------------------------------------------------------

def make_pc_background_knowledge(
    variable_names: list[str],
    forbidden_edges: list[tuple[str, str]],
    required_edges: list[tuple[str, str]],
) -> BackgroundKnowledge:
    """
    Convert generic forbidden/required edge lists into causal-learn BackgroundKnowledge.
    """
    bk = BackgroundKnowledge()
    nodes = {name: GraphNode(name) for name in variable_names}

    for source, target in forbidden_edges:
        bk.add_forbidden_by_node(nodes[source], nodes[target])

    for source, target in required_edges:
        bk.add_required_by_node(nodes[source], nodes[target])

    return bk


# ---------------------------------------------------------------------
# causal-learn graph conversion
# ---------------------------------------------------------------------

def causallearn_graph_to_edges(cg, variable_names: list[str]) -> pd.DataFrame:
    """
    Convert causal-learn graph matrix into a readable edge table.
    """
    mat = cg.G.graph
    rows = []

    for i in range(len(variable_names)):
        for j in range(i + 1, len(variable_names)):
            a = mat[i, j]
            b = mat[j, i]

            if a == 0 and b == 0:
                continue

            if a == -1 and b == 1:
                source = variable_names[i]
                target = variable_names[j]
                edge_type = "directed"

            elif a == 1 and b == -1:
                source = variable_names[j]
                target = variable_names[i]
                edge_type = "directed"

            elif a == -1 and b == -1:
                source = variable_names[i]
                target = variable_names[j]
                edge_type = "undirected"

            elif a == 1 and b == 1:
                source = variable_names[i]
                target = variable_names[j]
                edge_type = "undirected_or_partially_oriented"

            else:
                source = variable_names[i]
                target = variable_names[j]
                edge_type = f"unknown_encoding_{a}_{b}"

            rows.append({
                "source": source,
                "target": target,
                "edge_type": edge_type,
                "matrix_i_j": a,
                "matrix_j_i": b,
            })

    return pd.DataFrame(
        rows,
        columns=[
            "source",
            "target",
            "edge_type",
            "matrix_i_j",
            "matrix_j_i",
        ],
    )


# ---------------------------------------------------------------------
# PC runner
# ---------------------------------------------------------------------

def run_pc_discovery(
    X: np.ndarray,
    variable_names: list[str],
    alpha: float,
    background_knowledge: BackgroundKnowledge | None,
) -> pd.DataFrame:
    """
    Run PC and return an edge table.
    """
    cg = pc(
        data=X,
        alpha=alpha,
        indep_test="fisherz",
        stable=True,
        uc_rule=0,
        uc_priority=2,
        background_knowledge=background_knowledge,
        node_names=variable_names,
        show_progress=False,
    )

    return causallearn_graph_to_edges(cg, variable_names)


# ---------------------------------------------------------------------
# Constraint checks
# ---------------------------------------------------------------------

def find_forbidden_edge_violations(
    edges: pd.DataFrame,
    forbidden_edges: list[tuple[str, str]],
) -> pd.DataFrame:
    """
    Check whether any directed output edge is in the forbidden edge list.
    """
    if edges.empty:
        return pd.DataFrame(columns=edges.columns)

    forbidden = set(forbidden_edges)

    directed = edges[edges["edge_type"] == "directed"].copy()

    violations = directed[
        directed.apply(
            lambda row: (row["source"], row["target"]) in forbidden,
            axis=1,
        )
    ]

    return violations


def find_missing_required_edges(
    edges: pd.DataFrame,
    required_edges: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """
    Check whether required directed edges are present in the output.
    """
    if not required_edges:
        return []

    directed = edges[edges["edge_type"] == "directed"]

    present = set(zip(directed["source"], directed["target"]))

    return sorted(set(required_edges) - present)


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------

def add_graph_column(edges: pd.DataFrame, graph_name: str) -> pd.DataFrame:
    edges = edges.copy()
    edges.insert(0, "graph", graph_name)
    return edges


def save_edges(edges: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(path, index=False)
    print(f"Saved edges: {path}")


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
        undirected = graph_edges[graph_edges["edge_type"] != "directed"]

        directed_edges = [
            f"{row.source} -> {row.target}"
            for row in directed.itertuples(index=False)
        ]

        undirected_edges = [
            f"{row.source} -- {row.target}"
            for row in undirected.itertuples(index=False)
        ]

        rows.append({
            "graph": graph_name,
            "n_directed": len(directed_edges),
            "n_undirected": len(undirected_edges),
            "n_total": len(graph_edges),
            "directed_edges": "; ".join(directed_edges),
            "undirected_edges": "; ".join(undirected_edges),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Run specifications
# ---------------------------------------------------------------------

def make_run_specs(variable_names: list[str]) -> list[dict]:
    """
    Build run specs:
      1. unconstrained baseline
      2. one constrained run for each YAML file in constraints/
    """
    specs = [
        {
            "constraint_name": "unconstrained",
            "background_knowledge": None,
            "forbidden_edges": [],
            "required_edges": [],
        }
    ]

    for config in load_constraint_configs(CONSTRAINT_DIR):
        forbidden_edges = get_forbidden_edges(variable_names, config)
        required_edges = get_required_edges(variable_names, config)

        background_knowledge = make_pc_background_knowledge(
            variable_names=variable_names,
            forbidden_edges=forbidden_edges,
            required_edges=required_edges,
        )

        specs.append({
            "constraint_name": safe_label(config["name"]),
            "background_knowledge": background_knowledge,
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

    for alpha in ALPHAS:
        for spec in run_specs:
            graph_name = (
                f"pc_"
                f"alpha_{alpha_label(alpha)}_"
                f"constraint_{spec['constraint_name']}"
            )

            edges = run_pc_discovery(
                X=X,
                variable_names=variable_names,
                alpha=alpha,
                background_knowledge=spec["background_knowledge"],
            )

            print_edge_counts(graph_name, edges)

            if spec["constraint_name"] != "unconstrained":
                violations = find_forbidden_edge_violations(
                    edges=edges,
                    forbidden_edges=spec["forbidden_edges"],
                )

                missing_required = find_missing_required_edges(
                    edges=edges,
                    required_edges=spec["required_edges"],
                )

                print(f"Forbidden-edge violations: {len(violations)}")
                print(f"Missing required edges:     {len(missing_required)}")

                if not violations.empty:
                    print()
                    print("Forbidden-edge violations:")
                    print(violations[["source", "target", "edge_type"]].to_string(index=False))

                if missing_required:
                    print()
                    print("Missing required edges:")
                    for source, target in missing_required:
                        print(f"{source} -> {target}")

            graph_edges = add_graph_column(edges, graph_name)

            save_edges(
                graph_edges,
                GRAPH_CSV_DIR / f"{graph_name}_edges.csv",
            )

            all_edges.append(graph_edges)

    if not all_edges:
        raise RuntimeError("No PC runs were completed.")

    all_edges_df = pd.concat(all_edges, ignore_index=True)

    print()
    save_edges(
        all_edges_df,
        OUTPUT_DIR / "pc_all_edges.csv",
    )

    summary = make_edge_summary(all_edges_df)
    summary_path = OUTPUT_DIR / "pc_edge_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()