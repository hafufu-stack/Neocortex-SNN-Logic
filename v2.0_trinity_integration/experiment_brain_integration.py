import numpy as np
import matplotlib.pyplot as plt
import random
import os
import time as py_time 
from numba import cuda

import integration_genes as genes
import integration_kernels as kernels

# ==========================================
# 🧪 Experiment 17: The Trinity Brain 
# ==========================================

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)

# --- Config ---
SIM_TIME_PER_TRIAL = 100
TRIALS_PER_PHASE = 200 
DT = 0.5

# Debug Scale (Small & Fast)
N_H_DG = 2048
N_H_CA3 = 512
N_H_CA1 = 512
N_CX_IN = 512
N_BG_MSN = 512
N_OUT = 2      

# ★ TUNED PARAMETERS (Adrenaline)
BRIDGE_GAIN = 20.0       # ★ Massive Gain
DA_BASE = 0.05           
LR_BG = 0.15             
TARGET_FIRE_RATE = 0.10  # ★ Aim high (10%)
INHIBITION_GAIN = 10.0   # ★ Weak inhibition (Let them run)
BG_BIAS = 15.0           # ★ Positive Bias (Force Activation)
MAX_WEIGHT_SUM = 500.0   
TEACHING_CURRENT = 30.0  

class TrinityBrain:
    def __init__(self):
        print(f"🧠 Building Trinity Brain v6.3 (Adrenaline Fixed)...")
        print(f"   [DG: {N_H_DG}] [CX: {N_CX_IN}] [BG: {N_BG_MSN}]")
        
        # 1. Hippocampus
        self.h_dg = genes.generate_hc_params(N_H_DG, "GC")
        self.h_ca3 = genes.generate_hc_params(N_H_CA3, "CA3")
        self.h_ca1 = genes.generate_hc_params(N_H_CA1, "CA1")
        
        all_idx = np.arange(N_H_DG)
        np.random.shuffle(all_idx)
        n_each = int(N_H_DG * 0.1)
        self.ctx0_idx = all_idx[:n_each]
        self.ctx1_idx = all_idx[n_each:2*n_each]
        
        self.d_mf = self._to_gpu(*genes.generate_hc_connections(N_H_DG, N_H_CA3, 0.01, 35.0))
        self.d_rec = self._to_gpu(*genes.generate_hc_connections(N_H_CA3, N_H_CA3, 0.02, 2.0, True))
        self.d_sc = self._to_gpu(*genes.generate_hc_connections(N_H_CA3, N_H_CA1, 0.05, 10.0, False))
        
        self.d_h_dg = self._alloc_layer(N_H_DG, self.h_dg)
        self.d_h_ca3 = self._alloc_layer(N_H_CA3, self.h_ca3)
        self.d_h_ca1 = self._alloc_layer(N_H_CA1, self.h_ca1)
        self.d_ca3_ps = cuda.to_device(np.zeros(N_H_CA3, dtype=np.int32))

        # 2. Neocortex
        self.c_in = genes.generate_cx_params(N_CX_IN, "RS")
        self.d_c_in = self._alloc_layer(N_CX_IN, self.c_in)
        self.d_bridge_in = cuda.to_device(np.zeros(N_CX_IN, dtype=np.float32))
        
        # Bridge
        bridge_mask = np.zeros((N_H_CA1, N_CX_IN), dtype=np.float32)
        half_h = N_H_CA1 // 2
        half_c = N_CX_IN // 2
        
        block0 = (np.random.rand(half_h, half_c) < 0.05).astype(np.float32)
        bridge_mask[0:half_h, 0:half_c] = block0 * np.random.uniform(0.5, 1.5)
        
        block1 = (np.random.rand(N_H_CA1-half_h, N_CX_IN-half_c) < 0.05).astype(np.float32)
        bridge_mask[half_h:, half_c:] = block1 * np.random.uniform(0.5, 1.5)
        
        self.d_bridge_w = cuda.to_device(bridge_mask)

        # 3. Basal Ganglia
        self.bg_msn = genes.generate_bg_params(N_BG_MSN)
        self.d_bg_msn = self._alloc_layer(N_BG_MSN, self.bg_msn)
        
        # ★ Weights: Strong Init for Small Net
        w_cx_bg = np.zeros((N_CX_IN, N_BG_MSN), dtype=np.float32)
        d1_indices = np.concatenate([np.arange(0, 128), np.arange(256, 384)]) # Adjusted for 512 size
        d2_indices = np.concatenate([np.arange(128, 256), np.arange(384, 512)])
        
        # Init HUGE (0.5 - 0.8) to guarantee flow
        w_cx_bg[:, d1_indices] = np.random.uniform(0.5, 0.8, (N_CX_IN, len(d1_indices)))
        w_cx_bg[:, d2_indices] = np.random.uniform(0.1, 0.4, (N_CX_IN, len(d2_indices)))
        
        self.d_w_cx_bg = cuda.to_device(w_cx_bg)
        self.d_trace_cx_bg = cuda.to_device(np.zeros((N_CX_IN, N_BG_MSN), dtype=np.float32))
        
        # Correct Action Map for 512 neurons
        self.action_map = np.zeros(N_BG_MSN, dtype=np.int32)
        self.action_map[256:] = 1 
        self.d_action_map = cuda.to_device(self.action_map)
        
        # Correct Receptor Type for 512 neurons
        self.receptor_type = np.ones(N_BG_MSN, dtype=np.float32)
        self.receptor_type[128:256] = -1.0 
        self.receptor_type[384:512] = -1.0 
        self.d_receptor_type = cuda.to_device(self.receptor_type)
        
        self.d_pre_trace_cx = cuda.to_device(np.zeros(N_CX_IN, dtype=np.float32))

        # Dims
        self.dim_h = kernels.get_dims_1d(N_H_DG)
        self.dim_c = kernels.get_dims_1d(N_CX_IN)
        self.dim_bg = kernels.get_dims_1d(N_BG_MSN)
        self.dim_cx_bg = kernels.get_dims_2d(N_CX_IN, N_BG_MSN)
        self.dim_bridge = kernels.get_dims_1d(N_H_CA1)
        self.dim_bridge_2d = kernels.get_dims_2d(N_H_CA1, N_CX_IN)

    def _to_gpu(self, p, i, w):
        return (cuda.to_device(p), cuda.to_device(i), cuda.to_device(w))

    def _alloc_layer(self, n, p):
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

    def _run_hippocampus(self):
        kernels.update_neuron_kernel[self.dim_h[0], self.dim_h[1]](
            self.d_h_dg['v'], self.d_h_dg['u'], self.d_h_dg['a'], self.d_h_dg['b'], self.d_h_dg['c'], self.d_h_dg['d'],
            self.d_h_dg['i'], self.d_h_dg['s'], 0.0, DT, N_H_DG
        )
        kernels.clear_buffer_1d_kernel[self.dim_h[0], self.dim_h[1]](self.d_h_ca3['i'], N_H_CA3)
        kernels.sparse_synapse_kernel[self.dim_h[0], self.dim_h[1]](
            self.d_h_dg['s'], self.d_h_ca3['i'], *self.d_mf, N_H_DG
        )
        kernels.sparse_synapse_kernel[self.dim_h[0], self.dim_h[1]](
            self.d_ca3_ps, self.d_h_ca3['i'], *self.d_rec, N_H_CA3
        )
        kernels.update_neuron_kernel[self.dim_h[0], self.dim_h[1]](
            self.d_h_ca3['v'], self.d_h_ca3['u'], self.d_h_ca3['a'], self.d_h_ca3['b'], self.d_h_ca3['c'], self.d_h_ca3['d'],
            self.d_h_ca3['i'], self.d_h_ca3['s'], 0.0, DT, N_H_CA3
        )
        kernels.clear_buffer_1d_kernel[self.dim_h[0], self.dim_h[1]](self.d_h_ca1['i'], N_H_CA1)
        kernels.sparse_synapse_kernel[self.dim_h[0], self.dim_h[1]](
            self.d_h_ca3['s'], self.d_h_ca1['i'], *self.d_sc, N_H_CA3
        )
        kernels.update_neuron_kernel[self.dim_h[0], self.dim_h[1]](
            self.d_h_ca1['v'], self.d_h_ca1['u'], self.d_h_ca1['a'], self.d_h_ca1['b'], self.d_h_ca1['c'], self.d_h_ca1['d'],
            self.d_h_ca1['i'], self.d_h_ca1['s'], -2.0, DT, N_H_CA1
        )
        self.d_ca3_ps.copy_to_device(self.d_h_ca3['s'])

    def run_trial(self, context_idx, target_action, dopamine, is_testing=False):
        # ★ RESET 'u' (Fatigue) AND 'v' (Voltage) EVERY TRIAL
        for d in [self.d_h_dg, self.d_h_ca3, self.d_h_ca1, self.d_c_in, self.d_bg_msn]:
            d['v'].copy_to_device(np.full(d['v'].shape, -65.0, dtype=np.float32))
            d['u'].copy_to_device(np.zeros(d['u'].shape, dtype=np.float32))

        steps = int(SIM_TIME_PER_TRIAL / DT)
        bg_spike_counts = np.zeros(N_BG_MSN, dtype=np.int32)
        
        dg_input = np.zeros(N_H_DG, dtype=np.float32)
        if context_idx == 0: 
            half_dg = N_H_DG // 2
            dg_input[self.ctx0_idx[self.ctx0_idx < half_dg]] = 100.0 
        else:                
            half_dg = N_H_DG // 2
            dg_input[self.ctx1_idx[self.ctx1_idx >= half_dg]] = 100.0
            
        self.d_h_dg['i'].copy_to_device(dg_input)
        vis_stimulus = np.full(N_CX_IN, 20.0, dtype=np.float32) # ★ 20.0 to break threshold

        kernels.clear_buffer_2d_kernel[self.dim_cx_bg[0], self.dim_cx_bg[1]](self.d_trace_cx_bg, N_CX_IN, N_BG_MSN)
        kernels.clear_buffer_1d_kernel[self.dim_c[0], self.dim_c[1]](self.d_pre_trace_cx, N_CX_IN)

        for t in range(steps):
            self._run_hippocampus()
            
            # Bridge
            kernels.clear_buffer_1d_kernel[self.dim_c[0], self.dim_c[1]](self.d_bridge_in, N_CX_IN)
            kernels.bridge_matmul_kernel[self.dim_bridge[0], self.dim_bridge[1]](
                self.d_h_ca1['s'], self.d_bridge_w, self.d_bridge_in, N_H_CA1, N_CX_IN
            )
            cuda.synchronize()
            
            # Neocortex
            kernels.clear_buffer_1d_kernel[self.dim_c[0], self.dim_c[1]](self.d_c_in['i'], N_CX_IN)
            kernels.add_stimulus_kernel[self.dim_c[0], self.dim_c[1]](
                self.d_c_in['i'], self.d_bridge_in, vis_stimulus, BRIDGE_GAIN, N_CX_IN
            )
            kernels.update_neuron_kernel[self.dim_c[0], self.dim_c[1]](
                self.d_c_in['v'], self.d_c_in['u'], self.d_c_in['a'], self.d_c_in['b'], self.d_c_in['c'], self.d_c_in['d'],
                self.d_c_in['i'], self.d_c_in['s'], 0.0, DT, N_CX_IN
            )
            kernels.update_pre_trace_kernel[self.dim_c[0], self.dim_c[1]](
                self.d_c_in['s'], self.d_pre_trace_cx, 0.95, 1.0, N_CX_IN
            )

            # Basal Ganglia
            kernels.clear_buffer_1d_kernel[self.dim_bg[0], self.dim_bg[1]](self.d_bg_msn['i'], N_BG_MSN)
            kernels.dense_synapse_kernel[self.dim_cx_bg[0], self.dim_cx_bg[1]](
                self.d_c_in['s'], self.d_bg_msn['i'], self.d_w_cx_bg, N_CX_IN, N_BG_MSN
            )
            
            kernels.update_neuron_kernel[self.dim_bg[0], self.dim_bg[1]](
                self.d_bg_msn['v'], self.d_bg_msn['u'], self.d_bg_msn['a'], self.d_bg_msn['b'], self.d_bg_msn['c'], self.d_bg_msn['d'],
                self.d_bg_msn['i'], self.d_bg_msn['s'], BG_BIAS, DT, N_BG_MSN
            )
            
            # Teaching
            if not is_testing:
                kernels.apply_teaching_signal_kernel[self.dim_bg[0], self.dim_bg[1]](
                    self.d_bg_msn['i'], self.d_action_map, target_action, TEACHING_CURRENT, N_BG_MSN
                )
            
            bg_spikes = self.d_bg_msn['s'].copy_to_host()
            bg_spike_counts += bg_spikes
            
            # Inhibition
            current_fire_count = np.sum(bg_spikes)
            fire_frac = current_fire_count / N_BG_MSN
            
            if fire_frac > TARGET_FIRE_RATE:
                inh_amount = INHIBITION_GAIN * (fire_frac - TARGET_FIRE_RATE) * 10.0
                kernels.apply_global_inhibition_kernel[self.dim_bg[0], self.dim_bg[1]](
                    self.d_bg_msn['v'], inh_amount, N_BG_MSN
                )

            kernels.update_eligibility_trace_kernel[self.dim_cx_bg[0], self.dim_cx_bg[1]](
                self.d_bg_msn['s'], self.d_pre_trace_cx, self.d_trace_cx_bg, N_CX_IN, N_BG_MSN
            )
            
        return bg_spike_counts

def main():
    brain = TrinityBrain()
    print(f"🚀 Starting Trinity Learning (Adrenaline Fixed)...")
    
    history = []
    win_window = []
    phases = [
        ("Phase 1 (Ctx 0)", 0),
        ("Phase 2 (Ctx 1)", 1),
        ("Phase 3 (Mixed)", -1)
    ]
    
    # Reward Baseline
    reward_baseline = 0.0
    baseline_alpha = 0.01
    
    total_trial = 0
    for phase_name, fixed_ctx in phases:
        print(f"\n--- {phase_name} ---")
        is_testing = (fixed_ctx == -1)
        
        for i in range(TRIALS_PER_PHASE):
            ctx = fixed_ctx if fixed_ctx != -1 else np.random.randint(0, 2)
            target = ctx
            
            # Dopamine
            dopamine = DA_BASE 
            
            bg_spike_counts = brain.run_trial(ctx, target, 0.0, is_testing=is_testing)
            
            # Score
            act0_d1 = np.sum(bg_spike_counts[0:128]) 
            act0_d2 = np.sum(bg_spike_counts[128:256])
            act0_score = act0_d1 - act0_d2
            
            act1_d1 = np.sum(bg_spike_counts[256:384])
            act1_d2 = np.sum(bg_spike_counts[384:512])
            act1_score = act1_d1 - act1_d2
            
            if is_testing:
                action = 0 if act0_score > act1_score else 1
            else:
                scores = np.array([act0_score, act1_score], dtype=np.float32)
                # Normalize
                max_score = np.max(np.abs(scores))
                if max_score > 0:
                    scores = (scores / max_score) * 10.0 
                
                exp_s = np.exp(scores - np.max(scores))
                probs = exp_s / (np.sum(exp_s) + 1e-9)
                action = np.random.choice([0, 1], p=probs)
            
            is_correct = (action == target)
            reward = 5.0 if is_correct else -5.0
            
            reward_baseline = (1.0 - baseline_alpha) * reward_baseline + baseline_alpha * reward
            effective_dopamine = dopamine + (reward - reward_baseline)
            
            if not is_testing:
                kernels.dopamine_weight_update_kernel[brain.dim_cx_bg[0], brain.dim_cx_bg[1]](
                    brain.d_w_cx_bg, brain.d_trace_cx_bg, brain.d_receptor_type, 
                    effective_dopamine, LR_BG, N_CX_IN, N_BG_MSN
                )
                cuda.synchronize()
                
                # Normalize every 10 steps
                if i % 10 == 0:
                    kernels.normalize_weights_kernel[brain.dim_bg[0], brain.dim_bg[1]](
                        brain.d_w_cx_bg, N_CX_IN, N_BG_MSN, MAX_WEIGHT_SUM
                    )
                    cuda.synchronize()

            win_window.append(1 if is_correct else 0)
            if len(win_window) > 20: win_window.pop(0)
            acc = sum(win_window) / len(win_window)
            history.append(acc)
            
            total_trial += 1
            if i % 10 == 0:
                weights = brain.d_w_cx_bg.copy_to_host()
                w_mean = np.mean(weights)
                
                w_d1 = np.mean(weights[:, brain.receptor_type == 1.0])
                w_d2 = np.mean(weights[:, brain.receptor_type == -1.0])
                
                msn_fire = np.sum(bg_spike_counts)
                steps = int(SIM_TIME_PER_TRIAL / DT)
                rate = msn_fire / (N_BG_MSN * steps)
                
                try:
                    trace_sum = np.sum(brain.d_trace_cx_bg.copy_to_host())
                except:
                    trace_sum = -1.0
                
                print(f"Trial {total_trial:3d}: Ctx={ctx} | Res={'✅' if is_correct else '❌'} | Acc={acc:.2f} | Rate={rate:.3f} | Score0={act0_score} Score1={act1_score} | W_D1={w_d1:.3f} W_D2={w_d2:.3f} | Tr={trace_sum:.1e}")

    # ★ ROBUST PATH HANDLING FOR GRAPH SAVING
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.axhline(y=0.5, color='r', linestyle='--', label="Random Chance")
    
    plt.axvline(x=100, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=200, color='gray', linestyle=':', alpha=0.5)
    
    plt.title("Trinity Brain v2.0: Integrated Learning (HC-CX-BG)")
    plt.xlabel("Trial")
    plt.ylabel("Accuracy (Moving Average)")
    plt.legend()
    plt.xlim(0, len(history))
    plt.ylim(0, 1.05)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to find 'assets'
    assets_dir = os.path.join(os.path.dirname(script_dir), 'assets')
    
    if not os.path.exists(assets_dir):
        try:
            os.makedirs(assets_dir)
        except:
            # Fallback to current dir if permission denied or path issue
            assets_dir = script_dir
            
    save_path = os.path.join(assets_dir, "result_v2.0_trinity_success.png")
    
    plt.savefig(save_path)
    print(f"📊 Graph saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()