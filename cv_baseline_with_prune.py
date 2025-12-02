# cv_baseline_with_prune_fixed.py
# Usage:
#   pip install nuscenes-devkit numpy torch tqdm
#   edit NUSCENES_ROOT to your path
#   python cv_baseline_with_prune_fixed.py

import os, time, random
import numpy as np
import torch
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

# ---- config: set this to your local mini dataset folder ----
NUSCENES_ROOT = r"D:\PROGRAMMING\RESEARCH\AV-hazard\dataset"
NUSCENES_VERSION = "v1.0-mini"

H = 6   # future horizon (steps)
K = 5   # number of CV samples
DT = 0.5

# ---- safe nuScenes helpers ----
def safe_get(nusc, table_name, token):
    """Return record or None. Handles empty-string tokens and KeyError."""
    if not token:  # covers None and '' and empty
        return None
    try:
        return nusc.get(table_name, token)
    except Exception:
        return None

def get_ann_for_instance_in_sample(nusc, sample_token, instance_token):
    """
    Return sample_annotation or None. Safe wrapper: returns None on bad/missing tokens.
    """
    sample = safe_get(nusc, "sample", sample_token)
    if sample is None:
        return None
    for ann_t in sample.get('anns', []):
        ann = safe_get(nusc, 'sample_annotation', ann_t)
        if ann is None:
            continue
        if ann.get('instance_token') == instance_token:
            return ann
    return None

# ---- gather examples robustly ----
def gather_examples(nusc, max_scenes=200):
    samples = []
    for scene in nusc.scene[:max_scenes]:
        sample_token = scene.get('first_sample_token')
        # iterate sample chain safely
        while sample_token:
            sample = safe_get(nusc, 'sample', sample_token)
            if sample is None:
                break
            for ann_t in sample.get('anns', []):
                ann = safe_get(nusc, 'sample_annotation', ann_t)
                if ann is None:
                    continue
                inst = ann.get('instance_token')
                # previous sample's annotation
                prev_sample_token = sample.get('prev', None)
                prev_ann = get_ann_for_instance_in_sample(nusc, prev_sample_token, inst) if prev_sample_token else None
                if prev_ann is None:
                    continue
                # collect future H steps (stop if any missing)
                future = []
                cur_sample_token = sample_token
                valid_future = True
                for _ in range(H):
                    if not cur_sample_token:
                        valid_future = False
                        break
                    ann_future = get_ann_for_instance_in_sample(nusc, cur_sample_token, inst)
                    if ann_future is None:
                        valid_future = False
                        break
                    future.append(ann_future['translation'][:2])
                    # advance
                    cur_sample = safe_get(nusc, 'sample', cur_sample_token)
                    if cur_sample is None:
                        cur_sample_token = None
                    else:
                        cur_sample_token = cur_sample.get('next', None)
                if not valid_future:
                    continue
                pos_prev = prev_ann['translation'][:2]
                pos_curr = ann['translation'][:2]
                samples.append({
                    'scene_token': scene['token'],
                    'sample_token': sample_token,
                    'instance_token': inst,
                    'pos_prev': np.array(pos_prev, dtype=float),
                    'pos_curr': np.array(pos_curr, dtype=float),
                    'future_gt': np.array(future, dtype=float),
                })
            # next sample in chain
            sample_next = sample.get('next', None)
            if not sample_next:
                break
            sample_token = sample_next
    return samples

# ---- CV baseline predictor (unchanged) ----
def cv_predict(pos_prev, pos_curr, H=6, K=5, dt=0.5):
    vel = (pos_curr - pos_prev) / dt
    preds = np.zeros((K, H, 2), dtype=float)
    for k in range(K):
        jitter = np.random.randn(2) * 0.1
        v = vel + jitter
        for t in range(H):
            preds[k, t] = pos_curr + v * dt * (t+1)
    return preds

def ade_fde_min(gt, preds):
    d = np.linalg.norm(preds - gt[None,...], axis=2)
    ade = d.mean(axis=1)
    fde = d[:, -1]
    min_idx = np.argmin(ade)
    return float(ade[min_idx]), float(fde[min_idx])

# ---- attempt to import relevance mask, fallback to simple distance gate ----
try:
    from relevance_mask import compute_relevance_mask
except Exception:
    def compute_relevance_mask(agent_pos, edge_index, nusc_map, agent_speeds=None,
                               max_lane_hops=3, dist_threshold=50.0, tti_threshold=6.0,
                               default_speed=6.0, lane_radius=3.0):
        pos = np.asarray(agent_pos)
        E = edge_index.shape[1]
        mask = torch.zeros(E, dtype=torch.bool)
        src = edge_index[0].numpy().astype(int); dst = edge_index[1].numpy().astype(int)
        for e in range(E):
            i = src[e]; j = dst[e]
            if np.linalg.norm(pos[i]-pos[j]) <= dist_threshold:
                mask[e] = True
        return mask

# ---- main ----
if __name__ == "__main__":
    if not os.path.isdir(NUSCENES_ROOT):
        raise SystemExit(f"nuScenes folder not found: {NUSCENES_ROOT}")

    print("Loading nuScenes (this may take a few seconds)...")
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_ROOT, verbose=False)

    print("Collecting examples from nuScenes-mini (robust mode)...")
    examples = gather_examples(nusc, max_scenes=200)
    print("Examples collected:", len(examples))
    if len(examples) == 0:
        raise SystemExit("No examples found — check NUSCENES_ROOT and the dataset contents.")

    edges_before_total = 0
    edges_after_total = 0
    ades = []
    fdes = []
    latencies = []

    for ex in tqdm(examples[:300]):
        sample = safe_get(nusc, 'sample', ex['sample_token'])
        if sample is None:
            continue
        # build node positions from sample anns
        nodes = []
        for at in sample.get('anns', []):
            ann = safe_get(nusc, 'sample_annotation', at)
            if ann is None:
                continue
            nodes.append(ann['translation'][:2])
        node_pos = np.array(nodes, dtype=float)
        N = node_pos.shape[0]
        if N <= 1:
            continue
        src=[]; dst=[]
        for i in range(N):
            for j in range(N):
                if i==j: continue
                src.append(i); dst.append(j)
        edge_index = torch.LongTensor([src,dst])
        edges_before_total += edge_index.shape[1]

        # compute mask (safe: pass nusc.map or None)
        try:
            mask = compute_relevance_mask(node_pos, edge_index, nusc_map=nusc.map)
        except Exception:
            mask = compute_relevance_mask(node_pos, edge_index, None)
        keep_cnt = int(mask.sum().item())
        edges_after_total += keep_cnt

        # CV pred for this example's instance
        preds = cv_predict(ex['pos_prev'], ex['pos_curr'], H=H, K=K, dt=DT)
        ade, fde = ade_fde_min(ex['future_gt'], preds)
        ades.append(ade); fdes.append(fde)

    mean_ade = float(np.mean(ades)) if len(ades) else float('nan')
    mean_fde = float(np.mean(fdes)) if len(fdes) else float('nan')
    pruned_pct = 1.0 - (edges_after_total / float(edges_before_total + 1e-12))

    print("=== Results ===")
    print(f"Samples eval'd: {len(ades)}")
    print(f"minADE_{K} (CV baseline): {mean_ade:.4f} m")
    print(f"minFDE_{K} (CV baseline): {mean_fde:.4f} m")
    print(f"Edges before: {edges_before_total}, after: {edges_after_total}, pruned: {pruned_pct*100:.2f}%")
