import numpy as np
import math
from numba import cuda

# ==========================================
# 🧠 Integrated Kernels (v3.0 Trinity)
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

# --- Standard Neuron Update ---
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
            local_v = c[tid]
        
        if local_v > 30.0: local_v = 30.0
        if local_v < -90.0: local_v = -90.0
        
        v[tid] = local_v
        u[tid] = local_u
        spike_out[tid] = spiked

# --- BG Update with Homeostasis ---
@cuda.jit
def update_bg_neuron_kernel(v, u, th, a, b, c, d, I_input, spike_out, bias, dt, total_neurons):
    tid = cuda.grid(1)
    if tid < total_neurons:
        local_v = v[tid]
        local_u = u[tid]
        local_th = th[tid]
        
        # Adaptive Threshold Logic
        I = I_input[tid] + bias - local_th
        
        dv = 0.04 * local_v * local_v + 5.0 * local_v + 140.0 - local_u + I
        du = a[tid] * (b[tid] * local_v - local_u)
        
        local_v += dv * dt
        local_u += du * dt
        
        spiked = 0
        if local_v >= 30.0:
            local_v = c[tid]
            local_u += d[tid]
            spiked = 1
            local_th += 0.1 # Increase threshold if fired
        else:
            local_th *= 0.99 # Decay threshold if silent
        
        if local_v > 30.0: local_v = 30.0
        if local_v < -90.0: local_v = -90.0
        
        # Cap Threshold Bias
        if local_th > 30.0: local_th = 30.0
        if local_th < -10.0: local_th = -10.0

        v[tid] = local_v
        u[tid] = local_u
        th[tid] = local_th
        spike_out[tid] = spiked

# --- Proportional Inhibition ---
@cuda.jit
def apply_global_inhibition_kernel(v, inhibition_current, n_neurons):
    tid = cuda.grid(1)
    if tid < n_neurons:
        v[tid] -= inhibition_current
        if v[tid] < -90.0: v[tid] = -90.0

# --- Teaching Signal ---
@cuda.jit
def apply_teaching_signal_kernel(i_buffer, action_map, target_action, current_val, n_neurons):
    tid = cuda.grid(1)
    if tid < n_neurons:
        if action_map[tid] == target_action:
            i_buffer[tid] += current_val

# --- L2 Weight Normalization ---
@cuda.jit
def normalize_weights_l2_kernel(weights, n_pre, n_post, target_norm):
    col = cuda.grid(1)
    if col < n_post:
        sum_sq = 0.0
        for row in range(n_pre):
            w = weights[row, col]
            sum_sq += w * w
        
        norm = math.sqrt(sum_sq)
        
        if norm > 1e-6:
            scale = target_norm / norm
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

@cuda.jit
def update_eligibility_trace_kernel(post_spikes, pre_trace_cx, trace_cx_bg, n_pre, n_post):
    row, col = cuda.grid(2)
    if row < n_pre and col < n_post:
        val = trace_cx_bg[row, col] * 0.95
        if post_spikes[col] == 1 and pre_trace_cx[row] > 0.01:
            val += 0.2 * (pre_trace_cx[row] / 5.0) 
        if val > 5.0: val = 5.0
        trace_cx_bg[row, col] = val

@cuda.jit
def dopamine_weight_update_kernel(weights, traces, receptor_type, dopamine, learning_rate, n_pre, n_post):
    row, col = cuda.grid(2)
    if row < n_pre and col < n_post:
        tr = traces[row, col]
        if tr > 0.0:
            rec = receptor_type[col]
            rpe_magnitude = abs(dopamine)
            
            # Dynamic LR: Learn more from big surprises
            eff_lr = learning_rate * (1.0 + rpe_magnitude)
            
            # Moderate D2 Boost (3.0x)
            if rec < 0.0:
                eff_lr *= 3.0 
            
            delta = eff_lr * dopamine * rec * tr * 0.05
            weights[row, col] += delta
            traces[row, col] *= 0.6
            
        if weights[row, col] < 0.0: weights[row, col] = 0.0

# --- Cerebellum ---
@cuda.jit
def update_pf_trace_kernel(gc_spikes, pf_traces, n_gc, n_pc):
    row, col = cuda.grid(2)
    if row < n_gc and col < n_pc:
        val = pf_traces[row, col] * 0.9 
        if gc_spikes[row] == 1:
            val += 1.0
        if val > 10.0: val = 10.0
        pf_traces[row, col] = val

@cuda.jit
def cerebellum_ltd_kernel(weights, traces, error_signal, learning_rate, n_gc, n_pc):
    row, col = cuda.grid(2)
    if row < n_gc and col < n_pc:
        if error_signal[col] > 0.0:
            tr = traces[row, col]
            if tr > 0.0:
                delta = -1.0 * learning_rate * error_signal[col] * tr
                weights[row, col] += delta
        else:
             weights[row, col] += 0.0001 
        
        if weights[row, col] < 0.0: weights[row, col] = 0.0
        if weights[row, col] > 1.0: weights[row, col] = 1.0

@cuda.jit
def cerebellum_forward_kernel(gc_spikes, pc_input, weights, n_gc, n_pc):
    row, col = cuda.grid(2)
    if row < n_gc and col < n_pc:
        if gc_spikes[row] == 1:
            cuda.atomic.add(pc_input, col, weights[row, col])

# --- Thalamus ---
@cuda.jit
def update_thalamus_relay_kernel(raw_input, th_current, gating_signal, n_neurons):
    tid = cuda.grid(1)
    if tid < n_neurons:
        th_current[tid] = raw_input[tid] * gating_signal

# --- Decision STDP ---
@cuda.jit
def decision_stdp_kernel(bg_spikes, out_spikes, weights, learning_rate, n_bg, n_out):
    row, col = cuda.grid(2)
    if row < n_bg and col < n_out:
        if bg_spikes[row] == 1 and out_spikes[col] == 1:
            weights[row, col] += learning_rate * 0.1
            if weights[row, col] > 2.0: weights[row, col] = 2.0

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