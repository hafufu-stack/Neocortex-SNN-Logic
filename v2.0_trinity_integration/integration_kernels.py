import numpy as np
from numba import cuda

# ==========================================
# 🧠 Integrated Kernels 
# ==========================================

# --- Utility ---
@cuda.jit
def clear_buffer_1d_kernel(buffer, size):
    tid = cuda.grid(1)
    if tid < size: buffer[tid] = 0.0

@cuda.jit
def clear_buffer_2d_kernel(buffer, rows, cols):
    row, col = cuda.grid(2)
    if row < rows and col < cols:
        buffer[row, col] = 0.0

# --- Neuron Update ---
@cuda.jit
def update_neuron_kernel(v, u, a, b, c, d, I_input, spike_out, bias, dt, total_neurons):
    tid = cuda.grid(1)
    if tid < total_neurons:
        local_v = v[tid]
        local_u = u[tid]
        
        I = I_input[tid] + bias
        dv = 0.04 * local_v * local_v + 5.0 * local_v + 140.0 - local_u + I
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

# --- Proportional Inhibition ---
@cuda.jit
def apply_global_inhibition_kernel(v, inhibition_current, n_neurons):
    tid = cuda.grid(1)
    if tid < n_neurons:
        v[tid] -= inhibition_current
        if v[tid] < -90.0: v[tid] = -90.0

# --- Teaching Signal (Current) ---
@cuda.jit
def apply_teaching_signal_kernel(i_buffer, action_map, target_action, current_val, n_neurons):
    tid = cuda.grid(1)
    if tid < n_neurons:
        if action_map[tid] == target_action:
            i_buffer[tid] += current_val

# --- Weight Normalization ---
@cuda.jit
def normalize_weights_kernel(weights, n_pre, n_post, target_sum_max):
    col = cuda.grid(1)
    if col < n_post:
        current_sum = 0.0
        for row in range(n_pre):
            current_sum += weights[row, col]
        
        if current_sum > target_sum_max:
            scale = target_sum_max / current_sum
            for row in range(n_pre):
                weights[row, col] *= scale

# --- Hippocampus ---
@cuda.jit
def sparse_synapse_kernel(pre_spikes, post_current_buffer, syn_pointers, syn_indices, syn_weights, n_pre_neurons):
    tid = cuda.grid(1)
    if tid < n_pre_neurons:
        if pre_spikes[tid] == 1:
            start = syn_pointers[tid]
            end = syn_pointers[tid + 1]
            for k in range(start, end):
                target = syn_indices[k]
                w = syn_weights[k]
                cuda.atomic.add(post_current_buffer, target, w)

# --- Bridge ---
@cuda.jit
def bridge_matmul_kernel(ca1_spikes, bridge_w, c_in_buffer, n_ca1, n_cx):
    row = cuda.grid(1)
    if row < n_ca1:
        if ca1_spikes[row] == 1:
            for col in range(n_cx):
                w = bridge_w[row, col]
                if w > 0.0:
                    cuda.atomic.add(c_in_buffer, col, w)

@cuda.jit
def add_stimulus_kernel(c_in_i, bridge_in, vis_stim, gain, n_cx):
    tid = cuda.grid(1)
    if tid < n_cx:
        c_in_i[tid] = bridge_in[tid] * gain + vis_stim[tid]

# --- Traces ---
@cuda.jit
def update_pre_trace_kernel(pre_spikes, pre_traces, decay, incr, n_pre):
    tid = cuda.grid(1)
    if tid < n_pre:
        val = pre_traces[tid] * decay
        if pre_spikes[tid] == 1:
            val += 1.0
            if val > 5.0: val = 5.0
        pre_traces[tid] = val

# --- Basal Ganglia ---
@cuda.jit
def dense_synapse_kernel(pre_spikes, post_input, weights, n_pre, n_post):
    row, col = cuda.grid(2)
    if row < n_pre and col < n_post:
        if pre_spikes[row] == 1:
            cuda.atomic.add(post_input, col, weights[row, col])

# ★ GPT FIXED: Strict Trace Logic
@cuda.jit
def update_eligibility_trace_kernel(post_spikes, pre_trace_cx, trace_cx_bg, n_pre, n_post):
    row, col = cuda.grid(2)
    if row < n_pre and col < n_post:
        val = trace_cx_bg[row, col] * 0.95
        
        # Only bump if both Pre is active ( > 0.5) AND Post spikes
        if post_spikes[col] == 1 and pre_trace_cx[row] > 0.5:
            # Scaled increment (prevents explosion)
            val += 0.2 * (pre_trace_cx[row] / 5.0)
            
        if val > 5.0: val = 5.0
        trace_cx_bg[row, col] = val

# ★ GPT FIXED: Stable Weight Update
@cuda.jit
def dopamine_weight_update_kernel(weights, traces, receptor_type, dopamine, learning_rate, n_pre, n_post):
    row, col = cuda.grid(2)
    if row < n_pre and col < n_post:
        tr = traces[row, col]
        if tr > 0.0:
            rec = receptor_type[col]
            
            # Simple Biomimetic Rule:
            # DA > 0: D1+, D2-
            # DA < 0: D1-, D2+
            # No massive boost for D2 anymore
            delta = learning_rate * dopamine * rec * tr * 0.05
            weights[row, col] += delta
            
            # Consume trace
            traces[row, col] *= 0.5
            
        # Gentle passive decay
        weights[row, col] *= 0.9998
        
        # Allow zero weights (Full pruning)
        if weights[row, col] < 0.0: weights[row, col] = 0.0
        if weights[row, col] > 10.0: weights[row, col] = 10.0

# --- Dimensions ---
def get_dims_1d(n):
    tp = 256
    bg = (n + tp - 1) // tp
    return bg, tp

def get_dims_2d(rows, cols):
    tpx, tpy = 16, 16
    bgx = (rows + tpx - 1) // tpx
    bgy = (cols + tpy - 1) // tpy
    return (bgx, bgy), (tpx, tpy)