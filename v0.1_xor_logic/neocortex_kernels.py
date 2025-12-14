import numpy as np
from numba import cuda

# ==========================================
# 🧠 Neocortex Kernels
# ==========================================

@cuda.jit
def update_neuron_kernel(v, u, a, b, c, d, I_input, spike_out, dt, total_neurons):
    tid = cuda.grid(1)
    if tid < total_neurons:
        local_v = v[tid]
        local_u = u[tid]
        
        # Simple Izhikevich Update
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
    """
    Simple All-to-All connection for Logic Gates.
    weights matrix: [n_pre, n_post]
    """
    # 2D Grid for matrix multiplication
    row, col = cuda.grid(2)
    
    if row < n_pre and col < n_post:
        if pre_spikes[row] == 1:
            w = weights[row, col]
            # Atomic add to accumulator
            cuda.atomic.add(post_input, col, w)

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