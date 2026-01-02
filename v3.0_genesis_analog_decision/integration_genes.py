import numpy as np

# ==========================================
# 🧬 Integrated Genes - Fixed
# ==========================================

# --- 1. Hippocampus Parameters ---
HC_TYPES = {
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
    
    # Standard reset
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
        weights = np.random.uniform(0.1, weight_val, len(all_indices)).astype(np.float32)
    else:
        weights = np.full(len(all_indices), weight_val, dtype=np.float32)
        weights *= np.random.uniform(0.8, 1.2, len(weights))
    return pointers, all_indices, weights

# --- 2. Neocortex Parameters ---
def generate_cx_params(n_neurons):
    # RS (Regular Spiking)
    a = np.full(n_neurons, 0.02, dtype=np.float32)
    b = np.full(n_neurons, 0.2, dtype=np.float32)
    c = np.full(n_neurons, -65.0, dtype=np.float32)
    d = np.full(n_neurons, 8.0, dtype=np.float32)
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
    
    v = -60.0 * np.ones(n_neurons, dtype=np.float32)
    u = b * v
    
    # Adaptive Threshold (Homeostasis)
    thresh_bias = np.zeros(n_neurons, dtype=np.float32)
    
    return {"params": {"a": a, "b": b, "c": c, "d": d}, "state": {"v": v, "u": u, "th": thresh_bias}}

# --- 4. Cerebellum Parameters ---
def generate_cerebellum_params(n_gc, n_pc):
    gc_params = {
        "params": {
            "a": np.full(n_gc, 0.02, dtype=np.float32),
            "b": np.full(n_gc, 0.2, dtype=np.float32),
            "c": np.full(n_gc, -65.0, dtype=np.float32),
            "d": np.full(n_gc, 8.0, dtype=np.float32),
        },
        "state": {
            "v": -65.0 * np.ones(n_gc, dtype=np.float32),
            "u": 0.0 * np.ones(n_gc, dtype=np.float32)
        }
    }
    pc_params = {
        "params": {
            "a": np.full(n_pc, 0.1, dtype=np.float32), 
            "b": np.full(n_pc, 0.2, dtype=np.float32),
            "c": np.full(n_pc, -65.0, dtype=np.float32),
            "d": np.full(n_pc, 2.0, dtype=np.float32),
        },
        "state": {
            "v": -60.0 * np.ones(n_pc, dtype=np.float32), 
            "u": 0.0 * np.ones(n_pc, dtype=np.float32)
        }
    }
    return gc_params, pc_params

def generate_cerebellum_connections(n_gc, n_pc):
    weights = np.random.uniform(0.1, 0.3, (n_gc, n_pc)).astype(np.float32)
    traces = np.zeros((n_gc, n_pc), dtype=np.float32)
    return weights, traces

# --- 5. Thalamus Parameters ---
def generate_thalamus_params(n_neurons):
    a = np.full(n_neurons, 0.02, dtype=np.float32)
    b = np.full(n_neurons, 0.25, dtype=np.float32)
    c = np.full(n_neurons, -65.0, dtype=np.float32)
    d = np.full(n_neurons, 0.05, dtype=np.float32)
    v = -65.0 * np.ones(n_neurons, dtype=np.float32)
    u = b * v
    return {"params": {"a": a, "b": b, "c": c, "d": d}, "state": {"v": v, "u": u}}

# --- 6. Amygdala Parameters ---
def generate_amygdala_params(n_neurons):
    a = np.full(n_neurons, 0.02, dtype=np.float32)
    b = np.full(n_neurons, 0.2, dtype=np.float32)
    c = np.full(n_neurons, -65.0, dtype=np.float32)
    d = np.full(n_neurons, 8.0, dtype=np.float32)
    v = -65.0 * np.ones(n_neurons, dtype=np.float32)
    u = b * v
    return {"params": {"a": a, "b": b, "c": c, "d": d}, "state": {"v": v, "u": u}}

# --- 7. Decision Layer ---
def generate_decision_params(n_neurons):
    a = np.full(n_neurons, 0.02, dtype=np.float32)
    b = np.full(n_neurons, 0.2, dtype=np.float32)
    c = np.full(n_neurons, -65.0, dtype=np.float32)
    d = np.full(n_neurons, 8.0, dtype=np.float32)
    v = -65.0 * np.ones(n_neurons, dtype=np.float32)
    u = b * v
    return {"params": {"a": a, "b": b, "c": c, "d": d}, "state": {"v": v, "u": u}}

# --- Input Generators ---
def generate_structured_input(pattern_type, n_neurons, intensity=30.0):
    input_vector = np.zeros(n_neurons, dtype=np.float32)
    if pattern_type == 0:
        input_vector[::2] = intensity # Stripes
    elif pattern_type == 1:
        input_vector[:n_neurons//2] = intensity # Block
    
    input_vector += np.random.uniform(0, 5.0, n_neurons)
    return input_vector.astype(np.float32)