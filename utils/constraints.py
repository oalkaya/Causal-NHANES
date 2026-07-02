from __future__ import annotations

from pathlib import Path

import yaml


# ---------------------------------------------------------------------
# Optional YAML helpers
# ---------------------------------------------------------------------

def optional_list(config: dict, key: str) -> list:
    """
    Return a list for optional YAML fields.

    Handles:
      key missing
      key:
      key: []
    """
    value = config.get(key)

    if value is None:
        return []

    if not isinstance(value, list):
        raise ValueError(f"Expected {key!r} to be a list, got {type(value).__name__}.")

    return value


def optional_dict(config: dict, key: str) -> dict:
    """
    Return a dict for optional YAML fields.

    Handles:
      key missing
      key:
      key: {}
    """
    value = config.get(key)

    if value is None:
        return {}

    if not isinstance(value, dict):
        raise ValueError(f"Expected {key!r} to be a dict, got {type(value).__name__}.")

    return value


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_constraint_config(path: Path) -> dict:
    """
    Load one YAML constraint config.
    """
    if not path.exists():
        raise FileNotFoundError(f"Constraint config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Constraint config is empty: {path}")

    if "name" not in config:
        config["name"] = path.stem

    validate_constraint_config(config)

    return config


def load_constraint_configs(constraint_dir: Path) -> list[dict]:
    """
    Load all .yaml constraint configs in a folder.
    """
    if not constraint_dir.exists():
        raise FileNotFoundError(f"Constraint folder not found: {constraint_dir}")

    paths = sorted(constraint_dir.glob("*.yaml"))

    if not paths:
        raise FileNotFoundError(f"No .yaml files found in: {constraint_dir}")

    return [load_constraint_config(path) for path in paths]


# ---------------------------------------------------------------------
# Validation / parsing
# ---------------------------------------------------------------------

def validate_constraint_config(config: dict) -> None:
    """
    Basic sanity checks for one constraint config.
    """
    if "tiers" not in config:
        raise ValueError(f"Constraint config {config.get('name')} is missing 'tiers'.")

    make_tier_map(config)

    for edge in optional_list(config, "allowed_tier_breaks"):
        if "source" not in edge or "target" not in edge:
            raise ValueError(
                f"Invalid allowed_tier_breaks entry in {config['name']}: {edge}"
            )
        
    for edge in optional_list(config, "forbidden_edges"):
        if "source" not in edge or "target" not in edge:
            raise ValueError(
                f"Invalid forbidden_edges entry in {config['name']}: {edge}"
            )

    for edge in optional_list(config, "required_edges"):
        if "source" not in edge or "target" not in edge:
            raise ValueError(
                f"Invalid required_edges entry in {config['name']}: {edge}"
            )

    parent_restrictions = optional_dict(config, "parent_restrictions")

    for target, rule in parent_restrictions.items():
        if rule is None:
            raise ValueError(
                f"Parent restriction for {target} in {config['name']} is empty."
            )

        if "allowed_sources" not in rule:
            raise ValueError(
                f"Parent restriction for {target} in {config['name']} "
                "is missing 'allowed_sources'."
            )

        allowed_sources = rule["allowed_sources"]

        if allowed_sources is None:
            rule["allowed_sources"] = []

        elif not isinstance(allowed_sources, list):
            raise ValueError(
                f"Parent restriction for {target} in {config['name']} "
                "must have allowed_sources as a list."
            )


def make_tier_map(config: dict) -> dict[str, int]:
    """
    Convert YAML tiers into:
        variable_name -> tier_order
    """
    tier_map = {}

    for tier_name, tier_info in config["tiers"].items():
        if "order" not in tier_info:
            raise ValueError(f"Tier {tier_name!r} is missing 'order'.")

        if "variables" not in tier_info:
            raise ValueError(f"Tier {tier_name!r} is missing 'variables'.")

        order = tier_info["order"]
        variables = tier_info["variables"]

        if variables is None:
            variables = []

        if not isinstance(variables, list):
            raise ValueError(f"Tier {tier_name!r} variables must be a list.")

        for variable in variables:
            if variable in tier_map:
                raise ValueError(f"Variable appears in multiple tiers: {variable}")

            tier_map[variable] = order

    return tier_map


def check_constraint_variables(variable_names: list[str], config: dict) -> None:
    """
    Make sure all dataset variables are present in the YAML tiers.
    """
    tier_map = make_tier_map(config)

    missing = [name for name in variable_names if name not in tier_map]

    if missing:
        raise ValueError(
            f"Constraint config {config['name']!r} does not define tiers for: {missing}"
        )


# ---------------------------------------------------------------------
# Constraint logic
# ---------------------------------------------------------------------

def is_allowed_tier_break(source: str, target: str, config: dict) -> bool:
    """
    Return True if source -> target is an explicitly allowed tier-breaking edge.
    """
    for edge in optional_list(config, "allowed_tier_breaks"):
        if edge["source"] == source and edge["target"] == target:
            return True

    return False

def is_explicitly_forbidden(source: str, target: str, config: dict) -> bool:
    """
    Return True if source -> target is explicitly forbidden in YAML.
    """
    for edge in optional_list(config, "forbidden_edges"):
        if edge["source"] == source and edge["target"] == target:
            return True

    return False


def violates_parent_restriction(source: str, target: str, config: dict) -> bool:
    """
    Return True if target has an allowed-parent list and source is not in it.
    """
    parent_restrictions = optional_dict(config, "parent_restrictions")

    if target not in parent_restrictions:
        return False

    allowed_sources = parent_restrictions[target].get("allowed_sources") or []

    return source not in set(allowed_sources)


def allowed_edge(source: str, target: str, config: dict) -> bool:
    """
    Return True if source -> target is allowed.

    Rule order:
      1. No self loops.
      2. Explicit forbidden_edges block specific directed edges.
      3. Variables in no_parents cannot have incoming edges.
      4. Parent restrictions override tier breaks.
      5. Explicit allowed_tier_breaks can override the default tier rule.
      6. Later tiers cannot cause earlier tiers.
      7. Same-tier edges are allowed only if same_tier_edges_allowed is true.
    """
    if source == target:
        return False

    tier_map = make_tier_map(config)

    if source not in tier_map:
        raise ValueError(f"Source variable not found in tier config: {source}")

    if target not in tier_map:
        raise ValueError(f"Target variable not found in tier config: {target}")

    if is_explicitly_forbidden(source, target, config):
        return False

    no_parents = set(optional_list(config, "no_parents"))

    if target in no_parents:
        return False

    if violates_parent_restriction(source, target, config):
        return False

    source_tier = tier_map[source]
    target_tier = tier_map[target]

    if source_tier > target_tier:
        if is_allowed_tier_break(source, target, config):
            return True

        return False

    if source_tier == target_tier:
        return bool(config.get("same_tier_edges_allowed", True))

    return True


# ---------------------------------------------------------------------
# Edge extraction
# ---------------------------------------------------------------------

def get_required_edges(
    variable_names: list[str],
    config: dict,
) -> list[tuple[str, str]]:
    """
    Return required directed edges as:
        [(source, target), ...]
    """
    check_constraint_variables(variable_names, config)

    required_edges = []

    for edge in optional_list(config, "required_edges"):
        source = edge["source"]
        target = edge["target"]

        if source not in variable_names:
            raise ValueError(
                f"Required edge source {source!r} is not in dataset variables."
            )

        if target not in variable_names:
            raise ValueError(
                f"Required edge target {target!r} is not in dataset variables."
            )

        if source == target:
            raise ValueError(f"Self-loop cannot be required: {source} -> {target}")

        required_edges.append((source, target))

    return required_edges


def check_required_edges_are_allowed(
    variable_names: list[str],
    config: dict,
) -> None:
    """
    Required edges must not violate the forbidden-edge logic.
    """
    required_edges = get_required_edges(variable_names, config)

    for source, target in required_edges:
        if not allowed_edge(source, target, config):
            raise ValueError(
                f"Required edge violates constraints in {config['name']}: "
                f"{source} -> {target}"
            )


def get_forbidden_edges(
    variable_names: list[str],
    config: dict,
) -> list[tuple[str, str]]:
    """
    Return forbidden directed edges as:
        [(source, target), ...]
    """
    check_constraint_variables(variable_names, config)
    check_required_edges_are_allowed(variable_names, config)

    forbidden_edges = []

    for source in variable_names:
        for target in variable_names:
            if source == target:
                continue

            if not allowed_edge(source, target, config):
                forbidden_edges.append((source, target))

    return forbidden_edges


def get_allowed_edges(
    variable_names: list[str],
    config: dict,
) -> list[tuple[str, str]]:
    """
    Return allowed directed edges as:
        [(source, target), ...]
    """
    check_constraint_variables(variable_names, config)
    check_required_edges_are_allowed(variable_names, config)

    allowed_edges = []

    for source in variable_names:
        for target in variable_names:
            if source == target:
                continue

            if allowed_edge(source, target, config):
                allowed_edges.append((source, target))

    return allowed_edges


# ---------------------------------------------------------------------
# Optional summaries
# ---------------------------------------------------------------------

def summarize_constraints(variable_names: list[str], config: dict) -> dict:
    forbidden_edges = get_forbidden_edges(variable_names, config)
    allowed_edges = get_allowed_edges(variable_names, config)
    required_edges = get_required_edges(variable_names, config)

    return {
        "name": config["name"],
        "n_variables": len(variable_names),
        "n_allowed_edges": len(allowed_edges),
        "n_forbidden_edges": len(forbidden_edges),
        "n_required_edges": len(required_edges),
        "no_parents": optional_list(config, "no_parents"),
        "parent_restrictions": optional_dict(config, "parent_restrictions"),
        "allowed_tier_breaks": optional_list(config, "allowed_tier_breaks"),
        "required_edges": optional_list(config, "required_edges"),
        "forbidden_edges": optional_list(config, "forbidden_edges"),
    }


def print_constraint_summary(variable_names: list[str], config: dict) -> None:
    summary = summarize_constraints(variable_names, config)

    print()
    print("=" * 80)
    print(f"Constraint config: {summary['name']}")
    print("=" * 80)
    print(f"Variables:        {summary['n_variables']}")
    print(f"Allowed edges:    {summary['n_allowed_edges']}")
    print(f"Forbidden edges:  {summary['n_forbidden_edges']}")
    print(f"Required edges:   {summary['n_required_edges']}")

    if summary["no_parents"]:
        print(f"No-parent vars:   {summary['no_parents']}")

    if summary["parent_restrictions"]:
        print(f"Parent rules:     {summary['parent_restrictions']}")

    if summary["allowed_tier_breaks"]:
        print(f"Tier breaks:      {summary['allowed_tier_breaks']}")

    if summary["forbidden_edges"]:
        print(f"Explicit forbidden edges: {summary['forbidden_edges']}")

    if summary["required_edges"]:
        print(f"Required edges:   {summary['required_edges']}")
    
    