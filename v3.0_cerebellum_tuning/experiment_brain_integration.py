import numpy as np
import matplotlib.pyplot as plt
import random
import os
import time as py_time 
from numba import cuda

import integration_genes as genes
import integration_kernels as kernels

# ==========================================
# 🧪 Experiment 18: The Trinity Brain (v3.0 Final)
# ==========================================

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)

# --- Config ---
SIM_TIME_PER_TRIAL = 150
TRIALS_PER_PHASE = 200 
DT = 0.5

# Standard Scale (Stable)
N_H_DG = 16384 
N_H_CA3 = 4096
N_H_CA1 = 4096 
N_CX_IN = 4096   
N_BG_MSN = 4096 
N_OUT = 2      

# Missing Definitions
N_CB_GC = 8192   
N_CB_PC = 2      
N_TH = 4096      
N_AM = 1024      

# ★ TUNED PARAMETERS (Balanced)
BRIDGE_GAIN = 2.0        # Reduced to prevent saturation
DA_BASE = 0.05           
LR_BG = 0.05             
TARGET_FIRE_RATE = 0.05  # Sparse firing
INHIBITION_GAIN = 100.0  # Strong inhibition to stop seizures
BG_BIAS = 0.0            
MAX_WEIGHT_SUM = 20.0    # L2 Norm (Small value for vector length)
TEACHING_CURRENT = 20.0  
LR_OUT = 0.2             

class GenesisBrain:
    def __init__(self):
        print(f"🧠 Building Genesis Brain v3.0 (Final Release)...")
        print(f"   [DG:{N_H_DG}] [CX:{N_CX_IN}] [BG:{N_BG_MSN}]")
        
        # 1. Hippocampus
        self.h_dg = genes.generate_hc_params(N_H_DG, "GC")
        self.h_ca3 = genes.generate_hc_params(N_H_CA3, "CA3")
        self.h_ca1 = genes.generate_hc_params(N_H_CA1, "CA1")
        
        # Context Split
        half_dg = N_H_DG // 2
        n_each = int(N_H_DG * 0.1)
        self.ctx0_idx = np.random.choice(np.arange(0, half_dg), n_each, replace=False)
        self.ctx1_idx = np.random.choice(np.arange(half_dg, N_H_DG), n_each, replace=False)
        
        self.d_mf = self._to_gpu(*genes.generate_hc_connections(N_H_DG, N_H_CA3, 0.01, 35.0))
        self.d_rec = self._to_gpu(*genes.generate_hc_connections(N_H_CA3, N_H_CA3, 0.02, 2.0, True))
        self.d_sc = self._to_gpu(*genes.generate_hc_connections(N_H_CA3, N_H_CA1, 0.05, 10.0, False))
        
        self.d_h_dg = self._alloc_layer(N_H_DG, self.h_dg)
        self.d_h_ca3 = self._alloc_layer(N_H_CA3, self.h_ca3)
        self.d_h_ca1 = self._alloc_layer(N_H_CA1, self.h_ca1)
        self.d_ca3_ps = cuda.to_device(np.zeros(N_H_CA3, dtype=np.int32))

        # 2. Neocortex
        self.c_in = genes.generate_cx_params(N_CX_IN)
        self.d_c_in = self._alloc_layer(N_CX_IN, self.c_in)
        self.d_bridge_in = cuda.to_device(np.zeros(N_CX_IN, dtype=np.float32))
        
        # Visual Templates
        t0 = genes.generate_structured_input(0, N_CX_IN, intensity=30.0)
        t1 = genes.generate_structured_input(1, N_CX_IN, intensity=30.0)
        self.vis_templates = np.stack([t0, t1]).astype(np.float32)

        # Bridge
        bridge_mask = np.zeros((N_H_CA1, N_CX_IN), dtype=np.float32)
        half_h = N_H_CA1 // 2
        half_c = N_CX_IN // 2
        block0 = (np.random.rand(half_h, half_c) < 0.02).astype(np.float32)
        bridge_mask[0:half_h, 0:half_c] = block0 * np.random.uniform(0.1, 0.5)
        block1 = (np.random.rand(N_H_CA1-half_h, N_CX_IN-half_c) < 0.02).astype(np.float32)
        bridge_mask[half_h:, half_c:] = block1 * np.random.uniform(0.1, 0.5)
        self.d_bridge_w = cuda.to_device(bridge_mask)

        # 3. Basal Ganglia
        self.bg_msn = genes.generate_bg_params(N_BG_MSN)
        self.d_bg_msn = self._alloc_layer(N_BG_MSN, self.bg_msn)
        
        w_cx_bg = np.random.uniform(0.01, 0.05, (N_CX_IN, N_BG_MSN)).astype(np.float32)
        self.d_w_cx_bg = cuda.to_device(w_cx_bg)
        self.d_trace_cx_bg = cuda.to_device(np.zeros((N_CX_IN, N_BG_MSN), dtype=np.float32))
        
        self.action_map = np.zeros(N_BG_MSN, dtype=np.int32)
        self.action_map[2048:] = 1 
        self.d_action_map = cuda.to_device(self.action_map)
        
        self.receptor_type = np.ones(N_BG_MSN, dtype=np.float32)
        self.receptor_type[N_BG_MSN//2:] = -1.0 
        self.d_receptor_type = cuda.to_device(self.receptor_type)
        
        # 7. Decision Layer
        self.dec_out = genes.generate_decision_params(N_OUT)
        self.d_dec_out = self._alloc_layer(N_OUT, self.dec_out)
        self.d_w_bg_out = cuda.to_device(np.random.uniform(0.1, 0.5, (N_BG_MSN, N_OUT)).astype(np.float32))

        self.d_pre_trace_cx = cuda.to_device(np.zeros(N_CX_IN, dtype=np.float32))

        # 4. Cerebellum
        self.cb_gc, self.cb_pc = genes.generate_cerebellum_params(N_CB_GC, N_CB_PC)
        self.d_cb_gc = self._alloc_layer(N_CB_GC, self.cb_gc)
        self.d_cb_pc = self._alloc_layer(N_CB_PC, self.cb_pc)
        
        pf_w, pf_tr = genes.generate_cerebellum_connections(N_CB_GC, N_CB_PC)
        self.d_w_pf_pc = cuda.to_device(pf_w)
        self.d_trace_pf_pc = cuda.to_device(pf_tr)
        
        self.d_w_cx_gc = cuda.to_device(np.random.uniform(0.0, 1.0, (N_CX_IN, N_CB_GC)).astype(np.float32) * (5.0 / N_CX_IN))

        # 5. Thalamus
        self.th_tc = genes.generate_thalamus_params(N_TH)
        self.d_th_tc = self._alloc_layer(N_TH, self.th_tc)
        
        # 6. Amygdala
        self.am_la = genes.generate_amygdala_params(N_AM)
        self.d_am_la = self._alloc_layer(N_AM, self.am_la)

        # Dims
        self.dim_h = kernels.get_dims_1d(N_H_DG)
        self.dim_c = kernels.get_dims_1d(N_CX_IN)
        self.dim_bg = kernels.get_dims_1d(N_BG_MSN)
        self.dim_cx_bg = kernels.get_dims_2d(N_CX_IN, N_BG_MSN)
        self.dim_bridge = kernels.get_dims_1d(N_H_CA1)
        self.dim_bridge_2d = kernels.get_dims_2d(N_H_CA1, N_CX_IN)
        self.dim_gc = kernels.get_dims_1d(N_CB_GC)
        self.dim_pc = kernels.get_dims_1d(N_CB_PC)
        self.dim_cx_gc = kernels.get_dims_2d(N_CX_IN, N_CB_GC)
        self.dim_pf_pc = kernels.get_dims_2d(N_CB_GC, N_CB_PC)
        self.dim_th = kernels.get_dims_1d(N_TH)
        self.dim_am = kernels.get_dims_1d(N_AM)
        self.dim_out = kernels.get_dims_1d(N_OUT)
        self.dim_bg_out = kernels.get_dims_2d(N_BG_MSN, N_OUT)

    def _to_gpu(self, p, i, w):
        return (cuda.to_device(p), cuda.to_device(i), cuda.to_device(w))

    def _alloc_layer(self, n, p):
        data = {
            'v': cuda.to_device(p['state']['v']),
            'u': cuda.to_device(p['state']['u']),
            'a': cuda.to_device(p['params']['a']),
            'b': cuda.to_device(p['params']['b']),
            'c': cuda.to_device(p['params']['c']),
            'd': cuda.to_device(p['params']['d']),
            's': cuda.to_device(np.zeros(n, dtype=np.int32)),
            'i': cuda.to_device(np.zeros(n, dtype=np.float32))
        }
        if 'th' in p['state']:
            data['th'] = cuda.to_device(p['state']['th'])
        return data

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
        for d in [self.d_h_dg, self.d_h_ca3, self.d_h_ca1, self.d_c_in, self.d_bg_msn, self.d_cb_gc, self.d_cb_pc, self.d_th_tc, self.d_am_la, self.d_dec_out]:
            d['v'].copy_to_device(np.full(d['v'].shape, -65.0, dtype=np.float32))
            d['u'].copy_to_device(np.zeros(d['u'].shape, dtype=np.float32))

        steps = int(SIM_TIME_PER_TRIAL / DT)
        bg_spike_counts = np.zeros(N_BG_MSN, dtype=np.int32)
        out_spike_counts = np.zeros(N_OUT, dtype=np.int32)
        
        dg_input = np.zeros(N_H_DG, dtype=np.float32)
        if context_idx == 0: 
            half_dg = N_H_DG // 2
            dg_input[self.ctx0_idx[self.ctx0_idx < half_dg]] = 300.0 
        else:                
            half_dg = N_H_DG // 2
            dg_input[self.ctx1_idx[self.ctx1_idx >= half_dg]] = 300.0 
        self.d_h_dg['i'].copy_to_device(dg_input)
        
        # Visual Template
        if context_idx == -1:
             vis_stimulus = np.random.uniform(10.0, 30.0, N_CX_IN).astype(np.float32)
        else:
             vis_stimulus = self.vis_templates[context_idx]
        d_vis = cuda.to_device(vis_stimulus)

        kernels.clear_buffer_2d_kernel[self.dim_cx_bg[0], self.dim_cx_bg[1]](self.d_trace_cx_bg, N_CX_IN, N_BG_MSN)
        kernels.clear_buffer_1d_kernel[self.dim_c[0], self.dim_c[1]](self.d_pre_trace_cx, N_CX_IN)

        for t in range(steps):
            self._run_hippocampus()
            
            # --- Thalamus ---
            kernels.update_thalamus_relay_kernel[self.dim_th[0], self.dim_th[1]](d_vis, self.d_th_tc['i'], 1.0, N_TH)
            kernels.update_neuron_kernel[self.dim_th[0], self.dim_th[1]](
                self.d_th_tc['v'], self.d_th_tc['u'], self.d_th_tc['a'], self.d_th_tc['b'], self.d_th_tc['c'], self.d_th_tc['d'],
                self.d_th_tc['i'], self.d_th_tc['s'], 0.0, DT, N_TH
            )
            
            # --- Cortex ---
            kernels.clear_buffer_1d_kernel[self.dim_c[0], self.dim_c[1]](self.d_c_in['i'], N_CX_IN)
            kernels.bridge_matmul_kernel[self.dim_bridge[0], self.dim_bridge[1]](
                self.d_h_ca1['s'], self.d_bridge_w, self.d_c_in['i'], N_H_CA1, N_CX_IN
            )
            kernels.add_stimulus_kernel[self.dim_c[0], self.dim_c[1]](
                self.d_c_in['i'], self.d_th_tc['i'], self.d_th_tc['i'], BRIDGE_GAIN, N_CX_IN
            )
            kernels.update_neuron_kernel[self.dim_c[0], self.dim_c[1]](
                self.d_c_in['v'], self.d_c_in['u'], self.d_c_in['a'], self.d_c_in['b'], self.d_c_in['c'], self.d_c_in['d'],
                self.d_c_in['i'], self.d_c_in['s'], 0.0, DT, N_CX_IN
            )
            kernels.update_pre_trace_kernel[self.dim_c[0], self.dim_c[1]](
                self.d_c_in['s'], self.d_pre_trace_cx, 0.95, 1.0, N_CX_IN
            )

            # --- Basal Ganglia ---
            kernels.clear_buffer_1d_kernel[self.dim_bg[0], self.dim_bg[1]](self.d_bg_msn['i'], N_BG_MSN)
            kernels.dense_synapse_kernel[self.dim_cx_bg[0], self.dim_cx_bg[1]](
                self.d_c_in['s'], self.d_bg_msn['i'], self.d_w_cx_bg, N_CX_IN, N_BG_MSN
            )
            
            # Adaptive Threshold (Homeostasis)
            kernels.update_bg_neuron_kernel[self.dim_bg[0], self.dim_bg[1]](
                self.d_bg_msn['v'], self.d_bg_msn['u'], self.d_bg_msn['th'], 
                self.d_bg_msn['a'], self.d_bg_msn['b'], self.d_bg_msn['c'], self.d_bg_msn['d'],
                self.d_bg_msn['i'], self.d_bg_msn['s'], BG_BIAS, DT, N_BG_MSN
            )
            
            # Output Layer Update
            kernels.clear_buffer_1d_kernel[self.dim_out[0], self.dim_out[1]](self.d_dec_out['i'], N_OUT)
            kernels.dense_synapse_kernel[self.dim_bg_out[0], self.dim_bg_out[1]](
                self.d_bg_msn['s'], self.d_dec_out['i'], self.d_w_bg_out, N_BG_MSN, N_OUT
            )
            kernels.update_neuron_kernel[self.dim_out[0], self.dim_out[1]](
                self.d_dec_out['v'], self.d_dec_out['u'], self.d_dec_out['a'], self.d_dec_out['b'], self.d_dec_out['c'], self.d_dec_out['d'],
                self.d_dec_out['i'], self.d_dec_out['s'], 0.0, DT, N_OUT
            )

            bg_spikes = self.d_bg_msn['s'].copy_to_host()
            bg_spike_counts += bg_spikes
            out_spikes = self.d_dec_out['s'].copy_to_host()
            out_spike_counts += out_spikes
            
            # Inhibition
            current_fire_count = np.sum(bg_spikes)
            fire_frac = current_fire_count / N_BG_MSN
            if fire_frac > TARGET_FIRE_RATE * 2: 
                inh_amount = INHIBITION_GAIN * (fire_frac - TARGET_FIRE_RATE) * 10.0
                kernels.apply_global_inhibition_kernel[self.dim_bg[0], self.dim_bg[1]](
                    self.d_bg_msn['v'], inh_amount, N_BG_MSN
                )

            kernels.update_eligibility_trace_kernel[self.dim_cx_bg[0], self.dim_cx_bg[1]](
                self.d_bg_msn['s'], self.d_pre_trace_cx, self.d_trace_cx_bg, N_CX_IN, N_BG_MSN
            )
            
            # Teaching
            if not is_testing:
                kernels.apply_teaching_signal_kernel[self.dim_bg[0], self.dim_bg[1]](
                    self.d_bg_msn['i'], self.d_action_map, target_action, TEACHING_CURRENT, N_BG_MSN
                )
            
        return bg_spike_counts, out_spike_counts

def main():
    brain = GenesisBrain()
    print(f"🚀 Starting Genesis Learning (v3.0)...")
    
    history = []
    win_window = []
    phases = [
        ("Phase 1 (Ctx 0)", 0),
        ("Phase 2 (Ctx 1)", 1),
        ("Phase 3 (Mixed)", -1)
    ]
    
    reward_baseline = 0.0
    baseline_alpha = 0.01
    
    total_trial = 0
    for phase_name, fixed_ctx in phases:
        print(f"\n--- {phase_name} ---")
        is_testing = (fixed_ctx == -1)
        
        for i in range(TRIALS_PER_PHASE):
            ctx = fixed_ctx if fixed_ctx != -1 else np.random.randint(0, 2)
            target = ctx
            
            dopamine = DA_BASE 
            
            bg_spikes, out_spikes = brain.run_trial(ctx, target, 0.0, is_testing=is_testing)
            
            # Decision
            score0 = out_spikes[0]
            score1 = out_spikes[1]
            
            if is_testing:
                action = 0 if score0 > score1 else 1
                if score0 == score1: action = np.random.randint(0, 2)
            else:
                scores = np.array([score0, score1], dtype=np.float32)
                scores += np.random.uniform(0, 2.0, 2)
                action = np.argmax(scores)
            
            is_correct = (action == target)
            if is_correct:
                reward = 5.0
                kernels.decision_stdp_kernel[brain.dim_bg_out[0], brain.dim_bg_out[1]](
                    brain.d_bg_msn['s'], brain.d_dec_out['s'], brain.d_w_bg_out, LR_OUT, N_BG_MSN, N_OUT
                )
            else:
                reward = -2.0
                
            effective_dopamine = dopamine + (reward - reward_baseline)
            reward_baseline = (1.0 - baseline_alpha) * reward_baseline + baseline_alpha * reward
            
            if not is_testing:
                kernels.dopamine_weight_update_kernel[brain.dim_cx_bg[0], brain.dim_cx_bg[1]](
                    brain.d_w_cx_bg, brain.d_trace_cx_bg, brain.d_receptor_type, 
                    effective_dopamine, LR_BG, N_CX_IN, N_BG_MSN
                )
                cuda.synchronize()
                
                # ★ L2 Normalization (Every 10 steps)
                if i % 10 == 0:
                    kernels.normalize_weights_l2_kernel[brain.dim_bg[0], brain.dim_bg[1]](
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
                msn_fire = np.sum(bg_spikes)
                steps = int(SIM_TIME_PER_TRIAL / DT)
                rate = msn_fire / (N_BG_MSN * steps)
                
                try:
                    trace_sum = np.sum(brain.d_trace_cx_bg.copy_to_host())
                except:
                    trace_sum = -1.0
                
                print(f"Trial {total_trial:3d}: Ctx={ctx} | Res={'✅' if is_correct else '❌'} | Acc={acc:.2f} | Fire={msn_fire} | Rate={rate:.4f} | W_mean={w_mean:.4f} | Out={score0}vs{score1}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(os.path.dirname(current_dir), 'assets')
    if not os.path.exists(assets_dir):
        try: os.makedirs(assets_dir)
        except: assets_dir = current_dir
            
    save_path = os.path.join(assets_dir, "result_v3.0_genesis.png")
    
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.axhline(y=0.5, color='r', linestyle='--', label="Random Chance")
    plt.axvline(x=200, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=400, color='gray', linestyle=':', alpha=0.5)
    plt.title("Genesis Brain v3.0 (Experiment 18)")
    plt.xlabel("Trial")
    plt.ylabel("Accuracy")
    plt.savefig(save_path)
    print(f"📊 Graph saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()