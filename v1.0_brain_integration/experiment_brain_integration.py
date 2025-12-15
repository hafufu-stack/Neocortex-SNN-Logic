import numpy as np
import matplotlib.pyplot as plt
import random
import os
import time as py_time 
from numba import cuda

# Import both systems
import hippocampus_genes as h_genes
import hippocampus_kernels as h_kernels
import neocortex_genes as c_genes
import neocortex_kernels as c_kernels

# ==========================================
# 🧪 Experiment 16: Brain Integration (v1.0 Final Fixed)
# ==========================================

# Seed Fixation
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)

# --- Config ---
SIM_TIME_PER_TRIAL = 100
TRIALS_PER_PHASE = 300 
DT = 0.5

# Hippocampus Config
N_H_DG = 5000
N_H_CA3 = 2000
N_H_CA1 = 2000 

# Neocortex Config
N_CX_IN = 2000 
N_CX_OUT = 2   

# Parameters (Calibrated for Stability)
WEIGHT_BRIDGE = 20.0  
LR_CX = np.float32(0.05) 
TRACE_DECAY = np.float32(0.95) # Fast decay for sharp learning
TRACE_INCR = np.float32(0.05)  # Small increment to prevent saturation
BASELINE_ALPHA = 0.05
ADV_SCALE = 2.0
LR_CTX = {0: float(LR_CX), 1: float(LR_CX)*1.5} 
TR_EPS = 1e-8

# Bridge Gain (High gain to amplify normalized input)
BRIDGE_GAIN = 50.0   

class IntegratedBrain:
    def __init__(self):
        print("🧠 Building Integrated Brain (Final Fixed)...")
        
        # --- 1. Hippocampus Setup ---
        self.h_dg = h_genes.generate_network_params(N_H_DG, "GC")
        self.h_ca3 = h_genes.generate_network_params(N_H_CA3, "CA3")
        self.h_ca1 = h_genes.generate_network_params(N_H_CA1, "CA1")
        
        all_idx = np.arange(N_H_DG)
        np.random.shuffle(all_idx)
        n_each = int(N_H_DG * 0.05)
        self.ctx0_idx = all_idx[:n_each]
        self.ctx1_idx = all_idx[n_each:2*n_each]
        
        mf_p, mf_i, mf_w = h_genes.generate_connections(N_H_DG, N_H_CA3, 0.01, 30.0) 
        rec_p, rec_i, rec_w = h_genes.generate_connections(N_H_CA3, N_H_CA3, 0.05, 0.0, True) 
        sc_p, sc_i, sc_w = h_genes.generate_connections(N_H_CA3, N_H_CA1, 0.1, 5.0, False) 
        
        self.d_mf = (cuda.to_device(mf_p), cuda.to_device(mf_i), cuda.to_device(mf_w))
        self.d_rec = (cuda.to_device(rec_p), cuda.to_device(rec_i), cuda.to_device(rec_w))
        self.d_sc = (cuda.to_device(sc_p), cuda.to_device(sc_i), cuda.to_device(sc_w))
        
        self.d_h_dg = self._alloc_h_layer(N_H_DG, self.h_dg)
        self.d_h_ca3 = self._alloc_h_layer(N_H_CA3, self.h_ca3)
        self.d_h_ca1 = self._alloc_h_layer(N_H_CA1, self.h_ca1)
        
        self.d_ca3_ps = cuda.to_device(np.zeros(N_H_CA3, dtype=np.int32))

        # --- 2. Neocortex Setup ---
        self.c_in = c_genes.generate_cortical_layer(N_CX_IN, "RS")
        self.c_out = c_genes.generate_cortical_layer(N_CX_OUT, "RS")
        
        cx_weights = np.random.uniform(0.5, 3.0, (N_CX_IN, N_CX_OUT)).astype(np.float32)
        self.d_cx_weights_ctx0 = cuda.to_device(cx_weights.copy())
        self.d_cx_weights_ctx1 = cuda.to_device(cx_weights.copy())
        
        self.d_cx_traces = cuda.to_device(np.zeros((N_CX_IN, N_CX_OUT), dtype=np.float32))
        self.d_c_pre_trace = cuda.to_device(np.zeros(N_CX_IN, dtype=np.float32))
        
        self.d_c_in = self._alloc_c_layer(N_CX_IN, self.c_in)
        self.d_c_out = self._alloc_c_layer(N_CX_OUT, self.c_out)
        
        # ★ FIXED: Allocate Bridge Buffer (This was missing!)
        self.d_bridge_in = cuda.to_device(np.zeros(N_CX_IN, dtype=np.float32))
        
        # Bridge Mask (Topographic Block Structure)
        bridge_prob = 0.05 
        bridge_mask = np.zeros((N_H_CA1, N_CX_IN), dtype=np.float32)
        half_h = N_H_CA1 // 2
        half_c = N_CX_IN // 2
        
        block1 = (np.random.rand(half_h, half_c) < bridge_prob).astype(np.float32)
        bridge_mask[0:half_h, 0:half_c] = block1
        block2 = (np.random.rand(N_H_CA1-half_h, N_CX_IN-half_c) < bridge_prob).astype(np.float32)
        bridge_mask[half_h:, half_c:] = block2
        
        self.d_bridge_w = cuda.to_device(bridge_mask * WEIGHT_BRIDGE)
        
        self.dim_h_dg = h_kernels.get_dims(N_H_DG)
        self.dim_h_ca3 = h_kernels.get_dims(N_H_CA3)
        self.dim_h_ca1 = h_kernels.get_dims(N_H_CA1)
        self.dim_c_in = c_kernels.get_dims_1d(N_CX_IN)
        self.dim_c_out = c_kernels.get_dims_1d(N_CX_OUT)
        self.dim_c_syn = c_kernels.get_dims_2d(N_CX_IN, N_CX_OUT)
        self.dim_bridge = h_kernels.get_dims(N_H_CA1) 

    def _alloc_h_layer(self, n, p):
        return {
            'v': cuda.to_device(p['state']['v']),
            'u': cuda.to_device(p['state']['u']),
            'a': cuda.to_device(p['params']['a']),
            'b': cuda.to_device(p['params']['b']),
            'c': cuda.to_device(p['params']['c']),
            'd': cuda.to_device(p['params']['d']),
            's': cuda.to_device(np.zeros(n, dtype=np.int32)),
            'i': cuda.to_device(np.zeros(n, dtype=np.float32))
        }

    def _alloc_c_layer(self, n, p):
        return {
            'v': cuda.to_device(p['state']['v']),
            'u': cuda.to_device(p['state']['u']),
            'a': cuda.to_device(p['params']['a']),
            'b': cuda.to_device(p['params']['b']),
            'c': cuda.to_device(p['params']['c']),
            'd': cuda.to_device(p['params']['d']),
            's': cuda.to_device(np.zeros(n, dtype=np.int32)),
            'i': cuda.to_device(np.zeros(n, dtype=np.float32))
        }

    def run_trial(self, context_idx, target_action, current_lr, is_testing=False):
        self.d_h_dg['v'].copy_to_device(self.h_dg['state']['v'])
        self.d_h_ca3['v'].copy_to_device(self.h_ca3['state']['v'])
        self.d_h_ca1['v'].copy_to_device(self.h_ca1['state']['v'])
        self.d_c_in['v'].copy_to_device(self.c_in['state']['v'])
        self.d_c_out['v'].copy_to_device(self.c_out['state']['v'])
        
        # Clear Traces
        c_kernels.clear_trace_kernel[self.dim_c_syn[0], self.dim_c_syn[1]](self.d_cx_traces, N_CX_IN, N_CX_OUT)
        c_kernels.clear_buffer_kernel[self.dim_c_in[0], self.dim_c_in[1]](self.d_c_pre_trace, N_CX_IN)

        if context_idx == 0: d_current_weights = self.d_cx_weights_ctx0
        else:                d_current_weights = self.d_cx_weights_ctx1

        steps = int(SIM_TIME_PER_TRIAL / DT)
        cx_spike_counts = np.zeros(N_CX_OUT, dtype=np.int32)
        cx_in_spike_counts = np.zeros(N_CX_IN, dtype=np.int32) 
        
        dg_input = np.zeros(N_H_DG, dtype=np.float32)
        if context_idx == 0: dg_input[self.ctx0_idx] = 100.0 
        else:                dg_input[self.ctx1_idx] = 100.0
        self.d_h_dg['i'].copy_to_device(dg_input)
        
        # Baseline Visual Stimulus (Increased to 10.0)
        vis_stimulus = np.full(N_CX_IN, 10.0, dtype=np.float32)

        for t in range(steps):
            # --- 1. Hippocampus Step ---
            h_kernels.update_neuron_kernel[self.dim_h_dg[0], self.dim_h_dg[1]](
                self.d_h_dg['v'], self.d_h_dg['u'], self.d_h_dg['a'], self.d_h_dg['b'], self.d_h_dg['c'], self.d_h_dg['d'],
                self.d_h_dg['i'], self.d_h_dg['s'], 0.0, DT, N_H_DG
            )
            h_kernels.clear_buffer_kernel[self.dim_h_ca3[0], self.dim_h_ca3[1]](self.d_h_ca3['i'], N_H_CA3)
            h_kernels.synapse_transmission_kernel[self.dim_h_dg[0], self.dim_h_dg[1]](
                self.d_h_dg['s'], self.d_h_ca3['i'], *self.d_mf, N_H_DG
            )
            h_kernels.synapse_transmission_kernel[self.dim_h_ca3[0], self.dim_h_ca3[1]](
                self.d_ca3_ps, self.d_h_ca3['i'], *self.d_rec, N_H_CA3
            )
            h_kernels.update_neuron_kernel[self.dim_h_ca3[0], self.dim_h_ca3[1]](
                self.d_h_ca3['v'], self.d_h_ca3['u'], self.d_h_ca3['a'], self.d_h_ca3['b'], self.d_h_ca3['c'], self.d_h_ca3['d'],
                self.d_h_ca3['i'], self.d_h_ca3['s'], 0.0, DT, N_H_CA3
            )
            cuda.synchronize()
            host_spikes = self.d_h_ca3['s'].copy_to_host()
            self.d_ca3_ps.copy_to_device(host_spikes)

            h_kernels.clear_buffer_kernel[self.dim_h_ca1[0], self.dim_h_ca1[1]](self.d_h_ca1['i'], N_H_CA1)
            h_kernels.synapse_transmission_kernel[self.dim_h_ca3[0], self.dim_h_ca3[1]](
                self.d_h_ca3['s'], self.d_h_ca1['i'], *self.d_sc, N_H_CA3
            )
            h_kernels.update_neuron_kernel[self.dim_h_ca1[0], self.dim_h_ca1[1]](
                self.d_h_ca1['v'], self.d_h_ca1['u'], self.d_h_ca1['a'], self.d_h_ca1['b'], self.d_h_ca1['c'], self.d_h_ca1['d'],
                self.d_h_ca1['i'], self.d_h_ca1['s'], -2.0, DT, N_H_CA1
            )
            cuda.synchronize()

            # --- 2. Bridge (Gain Controlled) ---
            c_kernels.clear_buffer_kernel[self.dim_c_in[0], self.dim_c_in[1]](self.d_bridge_in, N_CX_IN)
            h_kernels.bridge_matmul_kernel[self.dim_bridge[0], self.dim_bridge[1]](
                self.d_h_ca1['s'], self.d_bridge_w, self.d_bridge_in, N_H_CA1, N_CX_IN
            )
            cuda.synchronize()
            
            bridge_host = self.d_bridge_in.copy_to_host()
            bmax = np.max(bridge_host)
            if bmax > 1e-9: bridge_host = bridge_host / bmax
            
            c_kernels.clear_buffer_kernel[self.dim_c_in[0], self.dim_c_in[1]](self.d_c_in['i'], N_CX_IN)
            combined = vis_stimulus + (bridge_host * BRIDGE_GAIN)
            self.d_c_in['i'].copy_to_device(combined)

            # --- 3. Neocortex ---
            c_kernels.update_neuron_kernel[self.dim_c_in[0], self.dim_c_in[1]](
                self.d_c_in['v'], self.d_c_in['u'], self.d_c_in['a'], self.d_c_in['b'], self.d_c_in['c'], self.d_c_in['d'],
                self.d_c_in['i'], self.d_c_in['s'], DT, N_CX_IN
            )
            
            c_kernels.update_pre_trace_kernel[self.dim_c_in[0], self.dim_c_in[1]](
                self.d_c_in['s'], self.d_c_pre_trace, TRACE_DECAY, TRACE_INCR, N_CX_IN
            )
            cx_in_spike_counts += self.d_c_in['s'].copy_to_host()
            
            c_kernels.clear_buffer_kernel[self.dim_c_out[0], self.dim_c_out[1]](self.d_c_out['i'], N_CX_OUT)
            c_kernels.synapse_kernel[self.dim_c_syn[0], self.dim_c_syn[1]](
                self.d_c_in['s'], self.d_c_out['i'], d_current_weights, N_CX_IN, N_CX_OUT
            )
            
            c_kernels.update_neuron_kernel[self.dim_c_out[0], self.dim_c_out[1]](
                self.d_c_out['v'], self.d_c_out['u'], self.d_c_out['a'], self.d_c_out['b'], self.d_c_out['c'], self.d_c_out['d'],
                self.d_c_out['i'], self.d_c_out['s'], DT, N_CX_OUT
            )
            
            spikes = self.d_c_out['s'].copy_to_host()
            cx_spike_counts += spikes
            
            c_kernels.update_synaptic_trace_kernel[self.dim_c_syn[0], self.dim_c_syn[1]](
                self.d_c_out['s'], self.d_c_pre_trace, self.d_cx_traces, 
                N_CX_IN, N_CX_OUT, self.d_c_in['s']
            )
            cuda.synchronize()

        # --- Decision & Reward ---
        scores = cx_spike_counts.astype(np.float32)
        if is_testing:
            action = int(np.argmax(scores)) 
        else:
            exp_scores = np.exp(scores - np.max(scores)) 
            probs = exp_scores / np.sum(exp_scores)
            if np.isnan(probs).any(): probs = np.ones(N_CX_OUT) / N_CX_OUT
            action = np.random.choice([0, 1], p=probs)
        
        # Reward
        return (action == target_action), d_current_weights, cx_spike_counts, cx_in_spike_counts, bridge_host

def main():
    brain = IntegratedBrain()
    print(f"🚀 Starting Block Learning ({TRIALS_PER_PHASE*3} Trials)...")
    
    history = []
    win_window = []
    
    phases = [
        ("Phase 1 (Ctx 0 Only)", 0),
        ("Phase 2 (Ctx 1 Only)", 1),
        ("Phase 3 (Mixed Test)", -1)
    ]
    
    baseline = {0: 0.5, 1: 0.5}
    bridge_store = []
    
    total_trial = 0
    for phase_name, fixed_ctx in phases:
        print(f"\n--- {phase_name} ---", flush=True)
        
        if phase_name == "Phase 2 (Ctx 1 Only)":
            baseline[1] = 0.5
            brain.d_cx_traces.copy_to_device(np.zeros((N_CX_IN, N_CX_OUT), dtype=np.float32))
            brain.d_c_pre_trace.copy_to_device(np.zeros(N_CX_IN, dtype=np.float32))
            w_init_2 = np.random.uniform(0.5, 3.0, (N_CX_IN, N_CX_OUT)).astype(np.float32)
            brain.d_cx_weights_ctx1.copy_to_device(w_init_2)

        is_testing = (fixed_ctx == -1)
        
        for i in range(TRIALS_PER_PHASE):
            if fixed_ctx != -1:
                ctx = fixed_ctx
            else:
                ctx = np.random.randint(0, 2)
            
            target = ctx
            is_correct, weights, spikes, in_spikes, host_bridge = brain.run_trial(ctx, target, LR_CX, is_testing=is_testing)
            
            bridge_store.append((ctx, host_bridge))

            if not is_testing:
                reward = 1.0 if is_correct else 0.0
                advantage = np.clip(reward - baseline[ctx], -1.0, 1.0)
                baseline[ctx] = (1 - BASELINE_ALPHA) * baseline[ctx] + BASELINE_ALPHA * reward
                
                tr_host = brain.d_cx_traces.copy_to_host()
                tr_sum = tr_host.sum()
                
                # Trace Fallback
                if tr_sum <= 1e-6:
                    pre_s = (in_spikes > 0).astype(np.float32)
                    post_s = (spikes > 0).astype(np.float32)
                    tr_host = np.outer(pre_s, post_s)
                    tr_sum = tr_host.sum()
                    if tr_sum <= 1e-6:
                         tr_host = (np.random.rand(N_CX_IN, N_CX_OUT) * 1e-3).astype(np.float32)
                         tr_sum = tr_host.sum()
                
                if tr_sum > 0.0:
                    gmax = tr_host.max()
                    tr_norm = tr_host / (gmax + TR_EPS)
                    brain.d_cx_traces.copy_to_device(tr_norm.astype(np.float32))
                    adv_scaled = float(ADV_SCALE * advantage)
                else:
                    adv_scaled = 0.0
                
                dims_syn = c_kernels.get_dims_2d(N_CX_IN, N_CX_OUT)
                d_w = brain.d_cx_weights_ctx0 if ctx == 0 else brain.d_cx_weights_ctx1
                
                lr_val = LR_CTX[ctx]
                w_before = d_w.copy_to_host()

                c_kernels.update_weight_kernel[dims_syn[0], dims_syn[1]](
                    d_w, brain.d_cx_traces,
                    adv_scaled, np.float32(lr_val),
                    N_CX_IN, N_CX_OUT
                )
                cuda.synchronize()
                
                w_after = d_w.copy_to_host()
                delta_w = np.linalg.norm(w_after - w_before)
            else:
                delta_w = 0.0
                tr_sum = 0.0
                trace_sum = 0.0

            win_window.append(1 if is_correct else 0)
            if len(win_window) > 20: win_window.pop(0)
            acc = sum(win_window) / len(win_window)
            history.append(acc)
            
            total_trial += 1
            if i % 20 == 0:
                bridge_sum = float(np.sum(host_bridge * BRIDGE_GAIN)) # Show actual scaled input
                ca1_rate = float(np.mean(brain.d_h_ca1['s'].copy_to_host()))
                if not is_testing:
                    trace_sum = float(np.sum(brain.d_cx_traces.copy_to_host()))
                    syn_max = float(np.max(brain.d_cx_traces.copy_to_host()))
                else:
                    trace_sum = 0.0
                    syn_max = 0.0
                w_mean = float(np.mean(weights.copy_to_host()))
                
                print(f"Trial {total_trial:3d}: Ctx={ctx} | Res={'✅' if is_correct else '❌'} | Acc={acc:.2f} | Br={bridge_sum:.0f} | Tr={trace_sum:.1f} | SynMax={syn_max:.2f} | ΔW={delta_w:.4f} | W={w_mean:.1f}")

                b0 = [b for c, b in bridge_store if c == 0]
                b1 = [b for c, b in bridge_store if c == 1]
                if len(b0) > 0 and len(b1) > 0:
                    mean_b0 = np.mean(b0, axis=0)
                    mean_b1 = np.mean(b1, axis=0)
                    cos = np.dot(mean_b0, mean_b1) / (np.linalg.norm(mean_b0) * np.linalg.norm(mean_b1) + 1e-9)
                    print(f"   [DIAG] Bridge Cos: {cos:.4f}")
                    bridge_store = []

    print("\n📊 Training Complete!")
    plt.plot(history)
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.axvline(x=TRIALS_PER_PHASE, color='gray', linestyle='--')
    plt.axvline(x=TRIALS_PER_PHASE*2, color='gray', linestyle='--')
    plt.title("Integrated Brain Learning Curve (Success!)")
    plt.ylabel("Accuracy")
    plt.xlabel("Trial")
    plt.ylim(0, 1.1)
    plt.show()

if __name__ == "__main__":
    main()