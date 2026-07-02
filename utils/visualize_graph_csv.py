from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.patches import FancyArrowPatch


# ---------------------------------------------------------------------
# Display layout
# ---------------------------------------------------------------------

NHANES_GROUPS = {
    0: [
        "Age",
        "Gender",
        "Income_Ratio",
    ],
    1: [
        "Vigorous_Activity",
        "Sedentary_Minutes",
        "Total_Calories",
        "Protein_g",
        "Carbohydrates_g",
        "Total_Sugars_g",
        "Dietary_Fiber_g",
    ],
    2: [
        "BMI",
        "Waist_Circumference",
        "Systolic_BP",
        "Diastolic_BP",
    ],
}

GROUP_LABELS = {
    0: "Demographics",
    1: "Behavior / Diet",
    2: "Cardiometabolic",
}

SHORT_NAMES = {
    "Age": "Age",
    "Gender": "Gender",
    "Income_Ratio": "Income",
    "BMI": "BMI",
    "Waist_Circumference": "Waist",
    "Vigorous_Activity": "Vigorous",
    "Sedentary_Minutes": "Sedentary",
    "Systolic_BP": "SysBP",
    "Diastolic_BP": "DiaBP",
    "Total_Calories": "Calories",
    "Protein_g": "Protein",
    "Carbohydrates_g": "Carbs",
    "Total_Sugars_g": "Sugars",
    "Dietary_Fiber_g": "Fiber",
}


# ---------------------------------------------------------------------
# Edge loading / normalization
# ---------------------------------------------------------------------

def load_edge_csv(path: Path) -> pd.DataFrame:
    edges = pd.read_csv(path)

    required = {"source", "target", "edge_type"}
    missing = required - set(edges.columns)

    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    edges = edges[["source", "target", "edge_type"]].copy()
    edges["edge_type"] = edges["edge_type"].astype(str).str.lower()

    return edges


def normalize_edge_type(edge_type: str) -> str:
    if edge_type == "directed":
        return "directed"

    if edge_type in {
        "undirected",
        "undirected_or_partially_oriented",
        "partially_oriented",
        "unresolved",
    }:
        return "undirected"

    if edge_type.startswith("unknown_encoding"):
        return "undirected"

    return "undirected"


# ---------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------

def build_graphs(edges: pd.DataFrame):
    directed_graph = nx.DiGraph()
    undirected_graph = nx.Graph()

    normalized_edges = []

    for _, row in edges.iterrows():
        source = row["source"]
        target = row["target"]
        edge_type = normalize_edge_type(row["edge_type"])

        normalized_edges.append((source, target, edge_type))

        directed_graph.add_node(source)
        directed_graph.add_node(target)
        undirected_graph.add_node(source)
        undirected_graph.add_node(target)

        if edge_type == "directed":
            directed_graph.add_edge(source, target)
        else:
            undirected_graph.add_edge(source, target)

    nodes = sorted(set(directed_graph.nodes()) | set(undirected_graph.nodes()))

    return directed_graph, undirected_graph, nodes, normalized_edges


def find_directed_cycle_edges(directed_graph: nx.DiGraph):
    cycles = list(nx.simple_cycles(directed_graph))
    cycle_edges = set()

    for cycle in cycles:
        for i in range(len(cycle)):
            source = cycle[i]
            target = cycle[(i + 1) % len(cycle)]
            cycle_edges.add((source, target))

    return cycles, cycle_edges


def find_undirected_cycles(undirected_graph: nx.Graph):
    return nx.cycle_basis(undirected_graph)


def save_cycle_report(
    output_base: Path,
    directed_cycles: list[list[str]],
    undirected_cycles: list[list[str]],
):
    rows = []

    for idx, cycle in enumerate(directed_cycles, start=1):
        rows.append({
            "cycle_type": "directed",
            "cycle_id": idx,
            "nodes": " -> ".join(cycle + [cycle[0]]),
        })

    for idx, cycle in enumerate(undirected_cycles, start=1):
        rows.append({
            "cycle_type": "undirected_unresolved",
            "cycle_id": idx,
            "nodes": " -- ".join(cycle + [cycle[0]]),
        })

    report = pd.DataFrame(rows)
    report_path = output_base.with_suffix(".cycles.csv")
    report.to_csv(report_path, index=False)

    return report_path, report


# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------

def make_grouped_layout(nodes: list[str]):
    groups = {k: list(v) for k, v in NHANES_GROUPS.items()}

    assigned = set()
    for group_nodes in groups.values():
        assigned.update(group_nodes)

    missing = [node for node in nodes if node not in assigned]

    if missing:
        groups[max(groups) + 1] = missing

    pos = {}

    x_gap = 3.6
    y_gap = 1.35

    for group_idx, group_nodes in groups.items():
        group_nodes = [node for node in group_nodes if node in nodes]

        n = len(group_nodes)

        for j, node in enumerate(group_nodes):
            x = group_idx * x_gap
            y = ((n - 1) / 2 - j) * y_gap
            pos[node] = (x, y)

    groups = {
        group_idx: [node for node in group_nodes if node in nodes]
        for group_idx, group_nodes in groups.items()
    }

    groups = {k: v for k, v in groups.items() if v}

    return pos, groups


def display_name(node: str) -> str:
    return SHORT_NAMES.get(node, node)


# ---------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------

def draw_graph(edges: pd.DataFrame, output_base: Path, title: str):
    directed_graph, undirected_graph, nodes, normalized_edges = build_graphs(edges)

    directed_cycles, directed_cycle_edges = find_directed_cycle_edges(directed_graph)
    undirected_cycles = find_undirected_cycles(undirected_graph)

    cycle_report_path, cycle_report = save_cycle_report(
        output_base=output_base,
        directed_cycles=directed_cycles,
        undirected_cycles=undirected_cycles,
    )

    raw_pos, groups = make_grouped_layout(nodes)

    pos = {
        display_name(node): xy
        for node, xy in raw_pos.items()
    }

    labels = {
        node: display_name(node)
        for node in nodes
    }

    directed_cycle_nodes = set()
    for cycle in directed_cycles:
        directed_cycle_nodes.update(cycle)

    fig, ax = plt.subplots(figsize=(18, 11), dpi=220)
    ax.set_title(title, fontsize=20, pad=18)
    ax.axis("off")

    # -----------------------------------------------------------------
    # Nodes
    # -----------------------------------------------------------------

    display_nodes = [labels[node] for node in nodes]

    node_edge_colors = [
        "red" if node in directed_cycle_nodes else "black"
        for node in nodes
    ]

    node_linewidths = [
        3.0 if node in directed_cycle_nodes else 1.5
        for node in nodes
    ]

    node_graph = nx.Graph()
    node_graph.add_nodes_from(display_nodes)

    nx.draw_networkx_nodes(
        node_graph,
        pos,
        node_size=3200,
        node_color="#f2f2f2",
        edgecolors=node_edge_colors,
        linewidths=node_linewidths,
        ax=ax,
    )

    nx.draw_networkx_labels(
        node_graph,
        pos,
        labels={labels[node]: labels[node] for node in nodes},
        font_size=12,
        font_weight="bold",
        ax=ax,
    )

    # -----------------------------------------------------------------
    # Edges
    # -----------------------------------------------------------------

    for source, target, edge_type in normalized_edges:
        source_label = labels[source]
        target_label = labels[target]

        if source_label not in pos or target_label not in pos:
            continue

        x1, y1 = pos[source_label]
        x2, y2 = pos[target_label]

        rad = 0.0

        # Curve same-column edges slightly.
        if abs(x1 - x2) < 1e-9:
            rad = 0.25 if y1 > y2 else -0.25

        if edge_type == "directed":
            in_cycle = (source, target) in directed_cycle_edges

            color = "red" if in_cycle else "#333333"
            linewidth = 3.2 if in_cycle else 2.1

            patch = FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=30,
                linewidth=linewidth,
                color=color,
                shrinkA=24,
                shrinkB=24,
                connectionstyle=f"arc3,rad={rad}",
                zorder=1,
            )

        else:
            # Undirected / unresolved edges.
            patch = FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-",
                linewidth=2.6,
                linestyle="--",
                color="#777777",
                shrinkA=24,
                shrinkB=24,
                connectionstyle=f"arc3,rad={rad}",
                zorder=1,
            )

        ax.add_patch(patch)

    # -----------------------------------------------------------------
    # Group labels
    # -----------------------------------------------------------------

    for group_idx, group_nodes in groups.items():
        display_group_nodes = [
            display_name(node)
            for node in group_nodes
            if display_name(node) in pos
        ]

        if not display_group_nodes:
            continue

        xs = [pos[node][0] for node in display_group_nodes]
        ys = [pos[node][1] for node in display_group_nodes]

        label = GROUP_LABELS.get(group_idx, f"Group {group_idx}")

        ax.text(
            xs[0],
            max(ys) + 1.0,
            label,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
        )

    # -----------------------------------------------------------------
    # Legend and status box
    # -----------------------------------------------------------------

    ax.plot([], [], color="#333333", linewidth=2.1, label="Directed edge")
    ax.plot([], [], color="#777777", linewidth=2.6, linestyle="--", label="Undirected / unresolved edge")
    ax.plot([], [], color="red", linewidth=3.2, label="Directed cycle edge")
    ax.legend(loc="lower right", fontsize=12, frameon=True)

    status_text = (
        f"Directed cycles: {len(directed_cycles)}\n"
        f"Undirected unresolved cycles: {len(undirected_cycles)}"
    )

    ax.text(
        0.01,
        0.01,
        status_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]

    ax.set_xlim(min(xs) - 1.8, max(xs) + 1.8)
    ax.set_ylim(min(ys) - 1.8, max(ys) + 1.8)

    fig.tight_layout()

    png_path = output_base.with_suffix(".png")
    svg_path = output_base.with_suffix(".svg")

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved PNG: {png_path}")
    print(f"Saved SVG: {svg_path}")
    print(f"Saved cycles: {cycle_report_path}")

    if not cycle_report.empty:
        print(cycle_report.to_string(index=False))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize all graph edge CSVs in a folder as PNG/SVG."
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Folder containing graph edge CSVs.",
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Folder where graph PNG/SVG files should be saved.",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(input_dir.glob("*.csv"))

    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    for csv_path in csv_paths:
        stem = csv_path.stem

        if stem.endswith("_edges"):
            stem = stem.removesuffix("_edges")

        title = stem.replace("_", " ")

        output_base = output_dir / stem

        print()
        print("=" * 80)
        print(f"Visualizing: {csv_path.name}")
        print("=" * 80)

        edges = load_edge_csv(csv_path)

        draw_graph(
            edges=edges,
            output_base=output_base,
            title=title,
        )


if __name__ == "__main__":
    main()