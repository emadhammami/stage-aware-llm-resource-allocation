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
