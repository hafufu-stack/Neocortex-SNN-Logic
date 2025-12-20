import numpy as np

# ==========================================
# 🧬 Integrated Genes 
# ==========================================

# --- 1. Hippocampus Parameters ---
HC_TYPES = {
    "EC":  {"a": 0.02, "b": 0.2,  "c": -65.0, "d": 8.0}, 
    "GC":  {"a": 0.02, "b": 0.2,  "c": -65.0, "d": 8.0}, 
    "CA3": {"a": 0.02, "b": 0.2,  "c": -55.0, "d": 4.0}, 
    "CA1": {"a": 0.02, "b": 0.2,  "c": -65.0, "d": 6.0}, 
}

def generate_hc_params(n_neurons, neuron_type="GC"):
    params = HC_TYPES[neuron_type]
    a = np.full(n_neurons, params["a"], dtype=np.float32)
    b = np.full(n_neurons, params["b"], dtype=np.float32)
    c = np.full(n_neurons, params["c"], dtype=np.float32)
    d = np.full(n_neurons, params["d"], dtype=np.float32)
    
    r = np.random.rand(n_neurons)
    c += 10.0 * r**2
    d -= 4.0 * r**2

    v = -65.0 * np.ones(n_neurons, dtype=np.float32)
    u = b * v
    return {"params": {"a": a, "b": b, "c": c, "d": d}, "state": {"v": v, "u": u}}

def generate_hc_connections(n_pre, n_post, connection_prob, weight_val, random_weight=False):
    pointers = [0]
    all_indices = []
    for i in range(n_pre):
        n_targets = int(n_post * connection_prob)
        if n_targets < 1: n_targets = 1
        targets = np.random.choice(n_post, n_targets, replace=False)
        all_indices.extend(targets)
        pointers.append(len(all_indices))
    all_indices = np.array(all_indices, dtype=np.int32)
    pointers = np.array(pointers, dtype=np.int32)
    
    if random_weight:
        # ★ Massive Weight Boost for Small Network
        weights = np.random.uniform(0.5, 1.5, len(all_indices)).astype(np.float32)
    else:
        weights = np.full(len(all_indices), weight_val, dtype=np.float32)
        weights *= np.random.uniform(0.8, 1.2, len(weights))
    return pointers, all_indices, weights

# --- 2. Neocortex Parameters ---
CX_TYPES = {
    "RS": {"a": 0.02, "b": 0.2,  "c": -65.0, "d": 8.0}, 
    "FS": {"a": 0.1,  "b": 0.2,  "c": -65.0, "d": 2.0}, 
}

def generate_cx_params(n_neurons, neuron_type="RS"):
    params = CX_TYPES[neuron_type]
    a = np.full(n_neurons, params["a"], dtype=np.float32)
    b = np.full(n_neurons, params["b"], dtype=np.float32)
    c = np.full(n_neurons, params["c"], dtype=np.float32)
    d = np.full(n_neurons, params["d"], dtype=np.float32)
    
    r = np.random.rand(n_neurons)
    if neuron_type == "RS":
        c += 15.0 * r**2
        d -= 6.0 * r**2
    
    v = -65.0 * np.ones(n_neurons, dtype=np.float32)
    u = b * v
    return {"params": {"a": a, "b": b, "c": c, "d": d}, "state": {"v": v, "u": u}}

# --- 3. Basal Ganglia Parameters ---
def generate_bg_params(n_neurons):
    # MSN: Striatum
    a = np.full(n_neurons, 0.02, dtype=np.float32)
    b = np.full(n_neurons, 0.2, dtype=np.float32)
    c = np.full(n_neurons, -65.0, dtype=np.float32)
    d = np.full(n_neurons, 8.0, dtype=np.float32) 
    
    v = -65.0 * np.ones(n_neurons, dtype=np.float32)
    u = b * v
    return {"params": {"a": a, "b": b, "c": c, "d": d}, "state": {"v": v, "u": u}}