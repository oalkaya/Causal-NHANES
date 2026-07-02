from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import pandas as pd


TREATMENTS = [
    "Vigorous_Activity",
    "Total_Calories",
    "Protein_g",
    "Carbohydrates_g",
]

OUTCOME = "BMI"


def load_directed_graph_from_edge_csv(path: Path) -> nx.DiGraph:
    """
    Load one graph edge CSV as a directed NetworkX graph.

    Expected columns:
        source
        target
        edge_type

    Only edge_type == "directed" is used.
    Undirected / partially oriented edges are ignored.
    """
    df = pd.read_csv(path)

    required_columns = {"source", "target", "edge_type"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    directed_edges = df[df["edge_type"] == "directed"]

    graph = nx.DiGraph()

    for row in directed_edges.itertuples(index=False):
        graph.add_edge(row.source, row.target)

    return graph


def get_directed_path(
    graph: nx.DiGraph,
    source: str,
    target: str,
) -> list[str] | None:
    """
    Return one directed path source -> ... -> target if it exists.
    Otherwise return None.
    """
    if source not in graph:
        return None

    if target not in graph:
        return None

    if not nx.has_path(graph, source, target):
        return None

    return nx.shortest_path(graph, source, target)


def check_graph_paths(
    path: Path,
    treatments: list[str],
    outcome: str,
) -> dict:
    """
    Check whether all treatments have directed paths to outcome.
    """
    graph = load_directed_graph_from_edge_csv(path)

    result = {
        "graph_csv": str(path),
        "graph_name": path.stem.removesuffix("_edges"),
        "is_dag": nx.is_directed_acyclic_graph(graph),
        "n_nodes": graph.number_of_nodes(),
        "n_directed_edges": graph.number_of_edges(),
    }

    all_paths_present = True

    for treatment in treatments:
        directed_path = get_directed_path(
            graph=graph,
            source=treatment,
            target=outcome,
        )

        has_path = directed_path is not None

        result[f"{treatment}_to_{outcome}_has_path"] = has_path
        result[f"{treatment}_to_{outcome}_path"] = (
            " -> ".join(directed_path) if directed_path else ""
        )

        if not has_path:
            all_paths_present = False

    result["all_treatments_have_path_to_outcome"] = all_paths_present

    return result


def check_graph_folder(
    graph_csv_dir: Path,
    treatments: list[str],
    outcome: str,
) -> pd.DataFrame:
    """
    Check all CSV graphs in a folder.
    """
    paths = sorted(graph_csv_dir.glob("*.csv"))

    if not paths:
        raise FileNotFoundError(f"No CSV files found in: {graph_csv_dir}")

    rows = []

    for path in paths:
        try:
            rows.append(
                check_graph_paths(
                    path=path,
                    treatments=treatments,
                    outcome=outcome,
                )
            )
        except Exception as exc:
            rows.append({
                "graph_csv": str(path),
                "graph_name": path.stem.removesuffix("_edges"),
                "error": str(exc),
                "all_treatments_have_path_to_outcome": False,
            })

    return pd.DataFrame(rows)


def print_compact_results(results: pd.DataFrame) -> None:
    columns = [
        "graph_name",
        "is_dag",
        "n_nodes",
        "n_directed_edges",
        "all_treatments_have_path_to_outcome",
    ]

    available_columns = [col for col in columns if col in results.columns]

    print()
    print("Compact path check:")
    print(results[available_columns].to_string(index=False))

    passing = results[
        results["all_treatments_have_path_to_outcome"] == True
    ]

    print()
    print(f"Graphs passing all treatment -> BMI path checks: {len(passing)}")

    if not passing.empty:
        print()
        for graph_name in passing["graph_name"]:
            print(f"- {graph_name}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "graph_csv_dir",
        type=Path,
        help="Folder containing graph edge CSV files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path for the path-check results.",
    )

    args = parser.parse_args()

    results = check_graph_folder(
        graph_csv_dir=args.graph_csv_dir,
        treatments=TREATMENTS,
        outcome=OUTCOME,
    )

    print_compact_results(results)

    output_path = args.output

    if output_path is None:
        output_path = args.graph_csv_dir.parent / "treatment_to_bmi_path_check.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()