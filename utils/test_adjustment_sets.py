import networkx as nx

from scripts.find_adjustment_sets import (
    find_minimal_adjustment_sets,
)


def run(graph, treatment="X", outcome="Y"):
    _, sets = find_minimal_adjustment_sets(
        graph=graph,
        treatment=treatment,
        outcome=outcome,
        max_set_size=None,
    )
    return {frozenset(s) for s in sets}


def test_simple_confounder():
    graph = nx.DiGraph([
        ("Z", "X"),
        ("Z", "Y"),
        ("X", "Y"),
    ])

    assert run(graph) == {frozenset({"Z"})}


def test_no_confounding():
    graph = nx.DiGraph([
        ("X", "Y"),
    ])

    assert run(graph) == {frozenset()}


def test_collider_requires_no_adjustment():
    graph = nx.DiGraph([
        ("X", "C"),
        ("Y", "C"),
    ])

    assert run(graph) == {frozenset()}


def test_two_confounders():
    graph = nx.DiGraph([
        ("Z1", "X"),
        ("Z1", "Y"),
        ("Z2", "X"),
        ("Z2", "Y"),
        ("X", "Y"),
    ])

    assert run(graph) == {
        frozenset({"Z1", "Z2"})
    }


def test_alternative_blockers():
    graph = nx.DiGraph([
        ("A", "X"),
        ("A", "B"),
        ("B", "Y"),
        ("X", "Y"),
    ])

    assert run(graph) == {
        frozenset({"A"}),
        frozenset({"B"}),
    }


def test_mediator_is_not_adjusted_for_total_effect():
    graph = nx.DiGraph([
        ("X", "M"),
        ("M", "Y"),
    ])

    assert run(graph) == {frozenset()}