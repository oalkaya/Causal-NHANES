from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd


def load_dag(graph_csv: Path) -> nx.DiGraph:
    if not graph_csv.exists():
        raise FileNotFoundError(f"Graph CSV not found: {graph_csv}")

    edges = pd.read_csv(graph_csv)

    required_columns = {"source", "target", "edge_type"}
    missing = required_columns - set(edges.columns)

    if missing:
        raise ValueError(
            f"Graph CSV is missing required columns: {sorted(missing)}"
        )

    graph = nx.DiGraph()

    # Add every node appearing in the CSV.
    graph.add_nodes_from(edges["source"].dropna().astype(str))
    graph.add_nodes_from(edges["target"].dropna().astype(str))

    directed_edges = edges[edges["edge_type"] == "directed"]

    graph.add_edges_from(
        zip(
            directed_edges["source"].astype(str),
            directed_edges["target"].astype(str),
        )
    )

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError(
            "The selected graph is not a DAG. "
            "Backdoor adjustment-set search requires a DAG."
        )

    return graph


def make_backdoor_graph(
    graph: nx.DiGraph,
    treatment: str,
) -> nx.DiGraph:
    """
    Construct the backdoor graph by removing every edge leaving treatment.

    After removing causal paths beginning at treatment, d-separation between
    treatment and outcome checks whether a set blocks all backdoor paths.
    """
    backdoor_graph = graph.copy()
    backdoor_graph.remove_edges_from(list(backdoor_graph.out_edges(treatment)))
    return backdoor_graph


def get_candidate_adjustment_variables(
    graph: nx.DiGraph,
    treatment: str,
    outcome: str,
) -> list[str]:
    """
    Restrict the search to relevant pre-treatment variables.

    Candidate variables are ancestors of the treatment or outcome, excluding:
      - treatment
      - outcome
      - descendants of treatment
    """
    ancestors = (
        nx.ancestors(graph, treatment)
        | nx.ancestors(graph, outcome)
    )

    descendants_of_treatment = nx.descendants(graph, treatment)

    candidates = (
        ancestors
        - descendants_of_treatment
        - {treatment, outcome}
    )

    return sorted(candidates)


def get_common_causes(
    graph: nx.DiGraph,
    treatment: str,
    outcome: str,
) -> list[str]:
    """
    Variables that are ancestors of both treatment and outcome.
    """
    return sorted(
        (
            nx.ancestors(graph, treatment)
            & nx.ancestors(graph, outcome)
        )
        - {treatment, outcome}
    )


def is_valid_adjustment_set(
    backdoor_graph: nx.DiGraph,
    treatment: str,
    outcome: str,
    adjustment_set: set[str],
) -> bool:
    """
    A valid set d-separates treatment and outcome in the backdoor graph.
    """
    try:
        # Current NetworkX API.
        return nx.is_d_separator(
            backdoor_graph,
            {treatment},
            {outcome},
            adjustment_set,
        )
    except AttributeError:
        # Compatibility with older NetworkX versions.
        return nx.d_separated(
            backdoor_graph,
            {treatment},
            {outcome},
            adjustment_set,
        )


def find_minimal_adjustment_sets(
    graph: nx.DiGraph,
    treatment: str,
    outcome: str,
    max_set_size: int | None,
) -> tuple[list[str], list[tuple[str, ...]]]:
    candidates = get_candidate_adjustment_variables(
        graph=graph,
        treatment=treatment,
        outcome=outcome,
    )

    backdoor_graph = make_backdoor_graph(
        graph=graph,
        treatment=treatment,
    )

    maximum_size = len(candidates)

    if max_set_size is not None:
        maximum_size = min(max_set_size, maximum_size)

    minimal_sets: list[tuple[str, ...]] = []

    # Search smaller sets first.
    for size in range(maximum_size + 1):
        for candidate_tuple in combinations(candidates, size):
            candidate_set = set(candidate_tuple)

            # If an already-found valid set is a subset, this larger set
            # cannot be inclusion-minimal.
            if any(
                set(existing).issubset(candidate_set)
                for existing in minimal_sets
            ):
                continue

            if is_valid_adjustment_set(
                backdoor_graph=backdoor_graph,
                treatment=treatment,
                outcome=outcome,
                adjustment_set=candidate_set,
            ):
                minimal_sets.append(candidate_tuple)

    return candidates, minimal_sets


def save_results(
    output_path: Path,
    treatment: str,
    outcome: str,
    common_causes: list[str],
    candidates: list[str],
    minimal_sets: list[tuple[str, ...]],
) -> None:
    rows = []

    for set_index, adjustment_set in enumerate(minimal_sets, start=1):
        rows.append({
            "treatment": treatment,
            "outcome": outcome,
            "set_index": set_index,
            "adjustment_set": "; ".join(adjustment_set),
            "n_adjustment_variables": len(adjustment_set),
            "common_causes": "; ".join(common_causes),
            "candidate_adjustment_variables": "; ".join(candidates),
        })

    if not rows:
        rows.append({
            "treatment": treatment,
            "outcome": outcome,
            "set_index": None,
            "adjustment_set": "",
            "n_adjustment_variables": None,
            "common_causes": "; ".join(common_causes),
            "candidate_adjustment_variables": "; ".join(candidates),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find inclusion-minimal valid backdoor adjustment sets "
            "for one treatment-outcome query."
        )
    )

    parser.add_argument(
        "graph_csv",
        type=Path,
        help="Graph edge CSV containing source, target and edge_type columns.",
    )

    parser.add_argument(
        "treatment",
        type=str,
        help="Treatment variable name.",
    )

    parser.add_argument(
        "outcome",
        type=str,
        help="Outcome variable name.",
    )

    parser.add_argument(
        "--max-set-size",
        type=int,
        default=None,
        help=(
            "Maximum adjustment-set size to search. "
            "Default: search all candidate-set sizes."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path.",
    )

    args = parser.parse_args()

    graph = load_dag(args.graph_csv)

    if args.treatment not in graph:
        raise ValueError(
            f"Treatment {args.treatment!r} is not present in the graph."
        )

    if args.outcome not in graph:
        raise ValueError(
            f"Outcome {args.outcome!r} is not present in the graph."
        )

    common_causes = get_common_causes(
        graph=graph,
        treatment=args.treatment,
        outcome=args.outcome,
    )

    candidates, minimal_sets = find_minimal_adjustment_sets(
        graph=graph,
        treatment=args.treatment,
        outcome=args.outcome,
        max_set_size=args.max_set_size,
    )

    has_causal_path = nx.has_path(
        graph,
        args.treatment,
        args.outcome,
    )

    causal_path = (
        nx.shortest_path(graph, args.treatment, args.outcome)
        if has_causal_path
        else []
    )

    print("=" * 80)
    print(f"Treatment: {args.treatment}")
    print(f"Outcome:   {args.outcome}")
    print("=" * 80)
    print(f"DAG nodes: {graph.number_of_nodes()}")
    print(f"DAG edges: {graph.number_of_edges()}")
    print(f"Directed causal path exists: {has_causal_path}")

    if causal_path:
        print(f"Example causal path: {' -> '.join(causal_path)}")

    print()
    print(f"Common causes: {common_causes}")
    print(f"Candidate adjustment variables: {candidates}")
    print()

    if minimal_sets:
        print(f"Minimal valid adjustment sets: {len(minimal_sets)}")

        for index, adjustment_set in enumerate(minimal_sets, start=1):
            formatted = ", ".join(adjustment_set) or "<empty set>"
            print(f"  {index}. {{{formatted}}}")
    else:
        print(
            "No valid adjustment set was found within the requested "
            "maximum set size."
        )

    output_path = args.output

    if output_path is None:
        graph_name = args.graph_csv.stem.removesuffix("_edges")
        filename = (
            f"{graph_name}"
            f"__{args.treatment}"
            f"__{args.outcome}"
            f"__minimal_adjustment_sets.csv"
        )

        output_path = (
            Path("outputs")
            / "adjustment_sets"
            / filename
        )

    save_results(
        output_path=output_path,
        treatment=args.treatment,
        outcome=args.outcome,
        common_causes=common_causes,
        candidates=candidates,
        minimal_sets=minimal_sets,
    )

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()