import numpy as np

# ==========================================
# 🧬 Neocortex Neuron Parameters
# ==========================================
# RS: Regular Spiking (Pyramidal Cell) - Main excitatory neuron
# FS: Fast Spiking (Interneuron) - Inhibitory, fires fast to block signals
NEURON_TYPES = {
    "RS": {"a": 0.02, "b": 0.2,  "c": -65.0, "d": 8.0}, 
    "FS": {"a": 0.1,  "b": 0.2,  "c": -65.0, "d": 2.0}, 
}

def generate_cortical_layer(n_neurons, neuron_type="RS"):
    """
    Generate parameters for a cortical layer.
    """
    params = NEURON_TYPES[neuron_type]
    
    a = np.full(n_neurons, params["a"], dtype=np.float32)
    b = np.full(n_neurons, params["b"], dtype=np.float32)
    c = np.full(n_neurons, params["c"], dtype=np.float32)
    d = np.full(n_neurons, params["d"], dtype=np.float32)
    
    # Slight heterogeneity
    r = np.random.rand(n_neurons)
    if neuron_type == "RS":
        c += 15.0 * r**2
        d -= 6.0 * r**2
    
    v = -65.0 * np.ones(n_neurons, dtype=np.float32)
    u = b * v

    return {"params": {"a": a, "b": b, "c": c, "d": d}, "state": {"v": v, "u": u}}