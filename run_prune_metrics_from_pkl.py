# run_prune_metrics_from_pkl.py
import pickle
import numpy as np

# -------- CONFIG --------
PREDICTIONS_PKL = r"D:\PROGRAMMING\university\masters\trajectory-prediction\Contrastive-Future-Trajectory-Prediction\outputs\nuScenes_VEHICLE_pred.pkl"
GROUND_TRUTH_PKL = r"D:\PROGRAMMING\university\masters\trajectory-prediction\dataset\v1.0-mini\sample_annotation.json"  # adjust if you have another GT
# Horizon & samples (optional, used in ADE/FDE)
H = 6
K = 5

# -------- HELPER FUNCTIONS --------
def compute_minADE_FDE(predictions, ground_truth):
    """
    predictions: list of dicts or array of shape [num_agents, K, H, 2]  # x,y
    ground_truth: dict of actual trajectories [num_agents, H, 2]
    returns: minADE_5, minFDE_5 in meters
    """
    ADE_list = []
    FDE_list = []
    for agent_id, gt_traj in ground_truth.items():
        pred_trajs = predictions.get(agent_id, None)
        if pred_trajs is None:
            continue
        # pred_trajs: [K,H,2]
        # compute L2 distance at each step
        l2 = np.linalg.norm(pred_trajs - gt_traj[None, :, :], axis=2)  # [K,H]
        ade_per_sample = l2.mean(axis=1)  # mean over H for each sample
        fde_per_sample = l2[:, -1]        # last step error
        ADE_list.append(ade_per_sample.min())  # take best sample
        FDE_list.append(fde_per_sample.min())
    minADE_5 = np.mean(ADE_list)
    minFDE_5 = np.mean(FDE_list)
    return minADE_5, minFDE_5

def relevance_mask(predictions, distance_threshold=30.0):
    """
    Simple relevance: drop edges where predicted distance > threshold
    Returns: mask dict {agent_id: [True/False ...]} indicating relevant connections
    """
    mask = {}
    for agent_id, pred_trajs in predictions.items():
        # compute distance to ego (0,0) for first step
        dist = np.linalg.norm(pred_trajs[:,0,:], axis=1)  # shape [K]
        mask[agent_id] = dist < distance_threshold
    return mask

def prune_edges(predictions, mask):
    edges_before = sum(len(v) for v in predictions.values())  # total samples
    edges_after = 0
    for agent_id, keep_mask in mask.items():
        edges_after += np.sum(keep_mask)
    pruned_pct = 100.0 * (edges_before - edges_after) / edges_before
    return edges_before, edges_after, pruned_pct

# -------- LOAD DATA --------
print("Loading predictions...")
with open(PREDICTIONS_PKL, "rb") as f:
    predictions = pickle.load(f)  # dict {agent_id: [K,H,2]}

# Load ground truth (example: sample_annotation.json)
import json
with open(GROUND_TRUTH_PKL, "r") as f:
    raw_gt = json.load(f)

# convert raw_gt to dict {agent_id: np.array([H,2])}
ground_truth = {}
for ann in raw_gt:
    agent_id = ann["instance_token"]
    # assuming future positions in ann["future_positions"]: [H,2]
    traj = np.array(ann["future_positions"])  # adapt if your JSON keys differ
    ground_truth[agent_id] = traj

# -------- RUN RELEVANCE PRUNE & METRICS --------
mask = relevance_mask(predictions, distance_threshold=30.0)
edges_before, edges_after, pruned_pct = prune_edges(predictions, mask)
minADE_5, minFDE_5 = compute_minADE_FDE(predictions, ground_truth)

# -------- PRINT RESULTS --------
print("=== Results ===")
print(f"Edges before: {edges_before}, after: {edges_after}, pruned: {pruned_pct:.2f}%")
print(f"minADE_5: {minADE_5:.4f} m")
print(f"minFDE_5: {minFDE_5:.4f} m")
