from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping, Protocol


class RouteNode(Protocol):
    def paths(self) -> tuple[frozenset[str], ...]: ...


@dataclass(frozen=True)
class Stage:
    stage_id: str

    def paths(self) -> tuple[frozenset[str], ...]:
        return (frozenset((self.stage_id,)),)


@dataclass(frozen=True)
class Sequence:
    children: tuple[RouteNode, ...]

    def paths(self) -> tuple[frozenset[str], ...]:
        if not self.children:
            return (frozenset(),)
        result: list[frozenset[str]] = []
        for combination in product(*(child.paths() for child in self.children)):
            merged: frozenset[str] = frozenset()
            for path in combination:
                merged = merged | path
            result.append(merged)
        return tuple(dict.fromkeys(result))


@dataclass(frozen=True)
class Exclusive:
    branches: tuple[RouteNode, ...]

    def paths(self) -> tuple[frozenset[str], ...]:
        return tuple(dict.fromkeys(path for branch in self.branches for path in branch.paths()))


def maximal_path(route: RouteNode, requirements: Mapping[str, int]) -> frozenset[str]:
    paths = route.paths()
    if not paths:
        return frozenset()
    return max(
        paths,
        key=lambda path: (sum(requirements[stage] for stage in path), tuple(sorted(path))),
    )


def maximal_requirement(route: RouteNode, requirements: Mapping[str, int]) -> int:
    path = maximal_path(route, requirements)
    return sum(requirements[stage] for stage in path)


def without_stages(route: RouteNode, excluded: set[str]) -> tuple[frozenset[str], ...]:
    return tuple(path - excluded for path in route.paths())


def maximal_remaining_path(
    route: RouteNode,
    requirements: Mapping[str, int],
    excluded: set[str],
) -> frozenset[str]:
    paths = without_stages(route, excluded)
    if not paths:
        return frozenset()
    return max(
        paths,
        key=lambda path: (sum(requirements[stage] for stage in path), tuple(sorted(path))),
    )

def topological_depths(route: RouteNode) -> dict[str, int]:
    depths: dict[str, int] = {}

    def visit(node: RouteNode, start: int) -> int:
        if isinstance(node, Stage):
            previous = depths.get(node.stage_id)
            depths[node.stage_id] = start if previous is None else min(previous, start)
            return 1
        if isinstance(node, Sequence):
            offset = 0
            for child in node.children:
                offset += visit(child, start + offset)
            return offset
        if isinstance(node, Exclusive):
            spans = [visit(branch, start) for branch in node.branches]
            return max(spans, default=0)
        raise TypeError(f"unsupported route node: {type(node)!r}")

    visit(route, 0)
    return depths


def optional_stages(route: RouteNode) -> frozenset[str]:
    paths = route.paths()
    if not paths:
        return frozenset()
    all_stages = set().union(*paths)
    unavoidable = set(paths[0])
    for path in paths[1:]:
        unavoidable.intersection_update(path)
    return frozenset(all_stages - unavoidable)
