# HiVT/utils/relevance_mask.py
import numpy as np
import torch
from collections import deque
from typing import Optional


def _closest_lane_token(nusc_map, x: float, y: float, radius: float = 3.0):
    try:
        return nusc_map.get_closest_lane(x, y, radius=radius)
    except Exception:
        return None


def _reachable_lanes_from(nusc_map, start_lane: str, max_hops: int = 3) -> set:
    if start_lane is None:
        return set()
    q = deque([(start_lane, 0)])
    visited = {start_lane}
    while q:
        lane, d = q.popleft()
        if d >= max_hops:
            continue
        out = nusc_map.get_outgoing_lanes(lane) or []
        for ol in out:
            if ol not in visited:
                visited.add(ol)
                q.append((ol, d + 1))
    return visited


def compute_relevance_mask(
    agent_pos: np.ndarray,
    edge_index: torch.LongTensor,
    nusc_map,
    agent_speeds: Optional[np.ndarray] = None,
    max_lane_hops: int = 3,
    dist_threshold: float = 50.0,
    tti_threshold: float = 6.0,
    default_speed: float = 6.0,
    lane_radius: float = 3.0,
) -> torch.BoolTensor:
    """
    Returns boolean mask (E,) True=keep edge.
    Rules:
      1) keep if lane connectivity (within max_lane_hops)
      2) else keep if distance <= dist_threshold AND tti <= tti_threshold
      3) else drop
    """
    assert isinstance(edge_index, torch.Tensor), "edge_index torch.Tensor[2,E]"
    pos = np.asarray(agent_pos)
    N = pos.shape[0]

    # lane tokens
    lane_tokens = [None] * N
    for i in range(N):
        lane_tokens[i] = _closest_lane_token(
            nusc_map, float(pos[i, 0]), float(pos[i, 1]), radius=lane_radius
        )

    unique = {lt for lt in lane_tokens if lt is not None}
    reachable_cache = {
        lt: _reachable_lanes_from(nusc_map, lt, max_hops=max_lane_hops) for lt in unique
    }

    if agent_speeds is None:
        speeds = np.full((N,), default_speed, dtype=float)
    else:
        speeds = np.asarray(agent_speeds).astype(float)
        speeds[speeds <= 0] = default_speed

    E = edge_index.shape[1]
    mask = torch.zeros(E, dtype=torch.bool, device=edge_index.device)

    srcs = edge_index[0].cpu().numpy().astype(int)
    dsts = edge_index[1].cpu().numpy().astype(int)

    for e in range(E):
        i = srcs[e]
        j = dsts[e]
        pi = pos[i]
        pj = pos[j]
        dist = float(np.linalg.norm(pi - pj))

        lane_i = lane_tokens[i]
        lane_j = lane_tokens[j]
        connected = False
        if lane_i is not None and lane_j is not None:
            if lane_i == lane_j:
                connected = True
            else:
                if lane_j in reachable_cache.get(
                    lane_i, set()
                ) or lane_i in reachable_cache.get(lane_j, set()):
                    connected = True

        if connected:
            mask[e] = True
            continue

        rel_speed = max(speeds[i], speeds[j], 1e-3)
        tti = dist / rel_speed
        if (dist <= dist_threshold) and (tti <= tti_threshold):
            mask[e] = True
        else:
            mask[e] = False

    return mask
