#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Dict, Any, Callable, Union, Iterable


def tree_clone_structure(default_value_factory: Callable, tree: Dict[str, Any]):
    coppied_tree = {}
    for k, v in tree.items():
        if isinstance(v, dict):
            coppied_tree[k] = tree_clone_structure(default_value_factory, v)
        else:
            coppied_tree[k] = default_value_factory()

    return coppied_tree


def tree_clone_shallow(tree: Dict[str, Any]):
    shallow_clone = {}
    for k, v in tree.items():
        if isinstance(v, dict):
            shallow_clone[k] = tree_clone_shallow(v)
        else:
            shallow_clone[k] = v

    return shallow_clone


def tree_append_in_place(target_tree: Dict[str, Any], source_tree: Dict[str, Any]):
    for k, v in source_tree.items():
        if isinstance(v, dict):
            target_tree[k] = tree_append_in_place(target_tree[k], v)
        else:
            target_tree[k].append(v)

    return target_tree


def tree_indexed_copy_in_place(
    target_tree: Dict[str, Any],
    source_tree: Dict[str, Any],
    target_index=None,
    source_index=None,
    non_blocking=False,
):
    for k, v in source_tree.items():
        if isinstance(v, dict):
            target_tree[k] = tree_indexed_copy_in_place(
                target_tree[k], v, target_index, source_index, non_blocking
            )
        else:
            if source_index is not None:
                v = v[source_index]

            tgt = target_tree[k]
            if target_index is not None:
                tgt = tgt[target_index]

            tgt.copy_(v, non_blocking=non_blocking)

    return target_tree


def tree_copy_in_place(
    target_tree: Dict[str, Any], source_tree: Dict[str, Any], non_blocking=False,
):
    return tree_indexed_copy_in_place(
        target_tree, source_tree, non_blocking=non_blocking
    )


def _tree_map_internal(
    func: Callable, target_tree: Dict[str, Any], source_tree: Dict[str, Any]
):
    for k, v in source_tree.items():
        if isinstance(v, dict):
            target_tree[k] = _tree_map_internal(func, target_tree.get(k, {}), v)
        else:
            target_tree[k] = func(v)

    return target_tree


def tree_map_in_place(func: Callable, tree: Dict[str, Any]):
    return _tree_map_internal(func, tree, tree)


def tree_map(func: Callable, tree: Dict[str, Any]):
    return _tree_map_internal(func, {}, tree)


def tree_select(inds, tree: Dict[str, Any]):
    return tree_map(lambda v: v[inds], tree)


def _tree_multi_map_internal(
    func: Callable,
    target_tree: Dict[str, Any],
    source_tree: Dict[str, Any],
    other_trees: List[Dict[str, Any]],
):
    for k, v in source_tree.items():
        if isinstance(v, dict):
            target_tree[k] = _tree_multi_map_internal(
                func, target_tree.get(k, {}), v, [t[k] for t in other_trees]
            )
        else:
            target_tree[k] = func(v, *(t[k] for t in other_trees))

    return target_tree


def tree_multi_map_in_place(func: Callable, tree: Dict[str, Any], *other_trees):
    return _tree_multi_map_internal(func, tree, tree, list(other_trees))


def tree_multi_map(func: Callable, tree: Dict[str, Any], *other_trees):
    return _tree_multi_map_internal(func, {}, tree, list(other_trees))
