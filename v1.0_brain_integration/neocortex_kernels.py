import numpy as np
from numba import cuda

# ==========================================
# 🧠 Neocortex Kernels (v1.0 Final Fix 8 - Sparse Trace)
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

# ★ FIX: Cap at 1.0
@cuda.jit
def update_pre_trace_kernel(pre_spikes, pre_traces, decay, incr, n_pre):
    tid = cuda.grid(1)
    if tid < n_pre:
        val = pre_traces[tid] * decay
        if pre_spikes[tid] == 1:
            val += incr
            if val > 1.0: val = 1.0 # Cap reduced from 10.0
        pre_traces[tid] = val

# ★ FIX: Thresholding & Low Cap
@cuda.jit
def update_synaptic_trace_kernel(
    post_spikes, pre_traces, synaptic_traces, 
    n_pre, n_post, pre_spikes 
):
    row, col = cuda.grid(2)
    if row < n_pre and col < n_post:
        # Threshold Check
        if pre_traces[row] > 0.05: # Only strong history matters
            add_val = 0.0
            
            # 1. Coincidence (Small bump)
            if pre_spikes[row] == 1 and post_spikes[col] == 1:
                add_val += 0.2

            # 2. Pre-Trace Ingestion (Small bump)
            if post_spikes[col] == 1:
                add_val += pre_traces[row] * 0.05

            if add_val != 0.0:
                cuda.atomic.add(synaptic_traces, (row, col), add_val)
                # Strict Cap
                if synaptic_traces[row, col] > 0.3:
                    synaptic_traces[row, col] = 0.3

# ★ FIX: Stronger Decay
@cuda.jit
def update_weight_kernel(
    weights, synaptic_traces, 
    adv, learning_rate, 
    n_pre, n_post
):
    row, col = cuda.grid(2)
    if row < n_pre and col < n_post:
        change = learning_rate * adv * synaptic_traces[row, col]
        weights[row, col] += change
        
        if weights[row, col] > 100.0: weights[row, col] = 100.0
        if weights[row, col] < 0.0:   weights[row, col] = 0.0
        
        # Stronger decay to prevent accumulation
        synaptic_traces[row, col] *= 0.3 

@cuda.jit
def clear_buffer_kernel(buffer, size):
    tid = cuda.grid(1)
    if tid < size: buffer[tid] = 0.0

@cuda.jit
def clear_trace_kernel(traces, n_pre, n_post):
    row, col = cuda.grid(2)
    if row < n_pre and col < n_post:
        traces[row, col] = 0.0

def get_dims_1d(n):
    tp = 256
    bg = (n + tp - 1) // tp
    return bg, tp

def get_dims_2d(rows, cols):
    tpx, tpy = 16, 16
    bgx = (rows + tpx - 1) // tpx
    bgy = (cols + tpy - 1) // tpy
    return (bgx, bgy), (tpx, tpy)