import numpy as np
from numba import cuda

# ==========================================
# 🧠 Neocortex Kernels (v0.2 R-STDP)
# ==========================================

@cuda.jit
def update_neuron_kernel(v, u, a, b, c, d, I_input, spike_out, dt, total_neurons):
    tid = cuda.grid(1)
    if tid < total_neurons:
        local_v = v[tid]
        local_u = u[tid]
        
        dv = 0.04 * local_v * local_v + 5.0 * local_v + 140.0 - local_u + I_input[tid]
        du = a[tid] * (b[tid] * local_v - local_u)
        
        local_v += dv * dt
        local_u += du * dt
        
        spiked = 0
        if local_v >= 30.0:
            local_v = c[tid]
            local_u += d[tid]
            spiked = 1
            
        if local_v > 30.0: local_v = 30.0
        if local_v < -90.0: local_v = -90.0
        
        v[tid] = local_v
        u[tid] = local_u
        spike_out[tid] = spiked

@cuda.jit
def synapse_kernel(pre_spikes, post_input, weights, n_pre, n_post):
    row, col = cuda.grid(2)
    if row < n_pre and col < n_post:
        if pre_spikes[row] == 1:
            cuda.atomic.add(post_input, col, weights[row, col])

# ★ NEW: Reinforcement Learning Kernel
@cuda.jit
def reward_learning_kernel(
    pre_spikes, post_spikes, 
    weights, traces,      # Traces: "Flag" that remembers recent activity
    reward,               # Global reward signal (+1, -1, or 0)
    learning_rate, 
    decay_factor,         # How fast the trace fades
    n_pre, n_post
):
    """
    R-STDP: 
    1. If Pre & Post fire, set Trace = 1.0 (Hebbian Tagging).
    2. If Reward arrives, update Weight based on Trace.
    3. Decay Trace over time.
    """
    row, col = cuda.grid(2)
    if row < n_pre and col < n_post:
        # 1. Update Eligibility Trace (Hebbian)
        # Simple Coincidence: If both fired recently
        if pre_spikes[row] == 1 and post_spikes[col] == 1:
            traces[row, col] = 1.0 # Set Flag!
            
        # 2. Apply Reward (Dopamine)
        if reward != 0.0:
            change = learning_rate * reward * traces[row, col]
            weights[row, col] += change
            
            # Clip weights (0.0 to 100.0)
            if weights[row, col] > 100.0: weights[row, col] = 100.0
            if weights[row, col] < 0.0:   weights[row, col] = 0.0
            
        # 3. Decay Trace
        # Trace fades away if no reward comes
        traces[row, col] *= decay_factor

@cuda.jit
def clear_buffer_kernel(buffer, size):
    tid = cuda.grid(1)
    if tid < size: buffer[tid] = 0.0

def get_dims_1d(n):
    tp = 256
    bg = (n + tp - 1) // tp
    return bg, tp

def get_dims_2d(rows, cols):
    tpx, tpy = 16, 16
    bgx = (rows + tpx - 1) // tpx
    bgy = (cols + tpy - 1) // tpy
    return (bgx, bgy), (tpx, tpy)