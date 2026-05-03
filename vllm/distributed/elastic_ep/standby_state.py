# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.distributed.parallel_state import (
    _init_stateless_group,
    _node_count,
    get_pp_group,
    get_tp_group,
    get_world_group,
)
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator

_STANDBY_WORLD: StatelessGroupCoordinator | None = None
_STANDBY_WORLD_NODE_COUNT: int | None = None
_STANDBY_DP: StatelessGroupCoordinator | None = None
_STANDBY_EP: StatelessGroupCoordinator | None = None
_STANDBY_EPLB: StatelessGroupCoordinator | None = None


def get_standby_dp_group() -> StatelessGroupCoordinator | None:
    return _STANDBY_DP


def get_standby_ep_group() -> StatelessGroupCoordinator | None:
    return _STANDBY_EP


def get_standby_eplb_group() -> StatelessGroupCoordinator | None:
    return _STANDBY_EPLB


def get_standby_world_group() -> StatelessGroupCoordinator | None:
    return _STANDBY_WORLD


def create_standby_groups(
    new_dp_size: int,
    new_world_size_across_dp: int,
    master_ip: str,
    coord_store_port: int,
    enable_eplb: bool = True,
    backend: str | None = None,
    survivor_old_dp_ranks: list[int] | None = None,
) -> None:
    """
    Create standby DP/EP/EPLB process groups.

    ``survivor_old_dp_ranks``: when provided (ungraceful-removal path),
    the standby group uses these OLD DP ranks instead of the default
    contiguous ``range(new_dp_size)``. Each surviving engine still
    presents its OLD ``world.rank`` to the StatelessGroupCoordinator,
    which then computes ``rank_in_group`` via ``ranks.index(world.rank)``.
    The default contiguous numbering only works when the survivors
    happen to be the first ``new_dp_size`` ranks (graceful scale-down
    drops trailing ranks). For an arbitrary dead set we must pass the
    real survivor ranks so each survivor finds itself in the list.
    """
    global \
        _STANDBY_WORLD, \
        _STANDBY_WORLD_NODE_COUNT, \
        _STANDBY_DP, \
        _STANDBY_EP, \
        _STANDBY_EPLB

    from vllm.distributed.utils import get_cached_tcp_store_client

    _td_world_size = torch.distributed.get_world_size()
    assert (
        new_world_size_across_dp == _td_world_size * new_dp_size
    ), (
        f"create_standby_groups arg mismatch: "
        f"new_world_size_across_dp={new_world_size_across_dp}, "
        f"torch.distributed.get_world_size()={_td_world_size}, "
        f"new_dp_size={new_dp_size}"
    )
    world_group = get_world_group()
    assert isinstance(world_group, StatelessGroupCoordinator), (
        f"world_group is {type(world_group).__name__}, "
        f"expected StatelessGroupCoordinator"
    )
    backend = backend or world_group.backend

    coord_store = get_cached_tcp_store_client(master_ip, coord_store_port)

    if survivor_old_dp_ranks is not None:
        # tp_size and pp_size are read below before they are known here;
        # but for our supported topology (TP=PP=PCP=1 in elastic EP) the
        # world rank == DP rank, so the survivor list is the ranks list
        # for both world and DP groups. (The general case would expand
        # each DP rank to the matching range of world ranks.)
        assert len(survivor_old_dp_ranks) == new_dp_size, (
            f"survivor_old_dp_ranks {survivor_old_dp_ranks} length must "
            f"equal new_dp_size {new_dp_size}"
        )
        standby_world_ranks = [list(survivor_old_dp_ranks)]
    else:
        standby_world_ranks = [list(range(new_world_size_across_dp))]
    _STANDBY_WORLD = _init_stateless_group(
        standby_world_ranks,
        "world",
        master_ip,
        backend,
        use_device_communicator=False,
        coord_store=coord_store,
    )
    _STANDBY_WORLD_NODE_COUNT = _node_count(_STANDBY_WORLD.tcp_store_group)

    tp_size = get_tp_group().world_size
    pp_size = get_pp_group().world_size

    if survivor_old_dp_ranks is not None:
        # Hard-remove path: TP/PP/PCP=1, so dp_ranks == world_ranks ==
        # ep_ranks. Build them all from the survivor list directly.
        assert tp_size == 1 and pp_size == 1, (
            "Hard-remove mode currently assumes TP=PP=1 (world rank == "
            "DP rank). Got tp_size=%d pp_size=%d." % (tp_size, pp_size)
        )
        standby_dp_ranks = [list(survivor_old_dp_ranks)]
        standby_ep_ranks = [list(survivor_old_dp_ranks)]
    else:
        all_ranks = torch.arange(new_world_size_across_dp).reshape(
            -1, new_dp_size, pp_size, tp_size
        )
        standby_dp_ranks = all_ranks.transpose(1, 3).reshape(
            -1, new_dp_size
        ).unbind(0)
        standby_dp_ranks = [x.tolist() for x in standby_dp_ranks]

        standby_ep_ranks = (
            all_ranks.transpose(1, 2).reshape(-1, new_dp_size * tp_size).unbind(0)
        )
        standby_ep_ranks = [x.tolist() for x in standby_ep_ranks]

    _STANDBY_DP = _init_stateless_group(
        standby_dp_ranks, "dp", master_ip, backend, coord_store=coord_store
    )
    _STANDBY_EP = _init_stateless_group(
        standby_ep_ranks, "ep", master_ip, backend, coord_store=coord_store
    )

    if enable_eplb:
        _STANDBY_EPLB = _init_stateless_group(
            standby_ep_ranks,
            "eplb",
            master_ip,
            backend,
            coord_store=coord_store,
        )


def pop_standby_groups() -> dict:
    """Return all standby groups and clear the standby state."""
    global \
        _STANDBY_WORLD, \
        _STANDBY_WORLD_NODE_COUNT, \
        _STANDBY_DP, \
        _STANDBY_EP, \
        _STANDBY_EPLB

    result = dict(
        world=_STANDBY_WORLD,
        dp=_STANDBY_DP,
        ep=_STANDBY_EP,
        eplb=_STANDBY_EPLB,
        node_count=_STANDBY_WORLD_NODE_COUNT,
    )
    _STANDBY_WORLD = None
    _STANDBY_WORLD_NODE_COUNT = None
    _STANDBY_DP = None
    _STANDBY_EP = None
    _STANDBY_EPLB = None
    return result
