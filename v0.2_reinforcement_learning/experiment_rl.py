import numpy as np
import matplotlib.pyplot as plt
import random
import os
from numba import cuda
import neocortex_genes
import neocortex_kernels

# ==========================================
# 🧪 Experiment 2: Reinforcement Learning (v0.2 Final Fix 3)
# ==========================================
# Improvement:
# - Boost Input Current (100 -> 300) to ensure firing
# - Keep Pre-Trace across trials (optional, but good for continuity)

# --- 1. Seed Fixation ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Config
SIM_TIME_PER_TRIAL = 100 
TRIALS = 300            
DT = 0.5

# Params
LEARNING_RATE = 0.05    
LR_DECAY = 0.999        
MIN_LR = 0.001          
BASELINE_ALPHA = 0.02   

# Trace Dynamics
TRACE_DECAY = 0.995     
TRACE_INCR = 0.1        
TRACE_MAX = 1.0

# Softmax Temperature
TEMP_START = 5.0        
TEMP_END = 0.1          

class RLCircuit:
    def __init__(self):
        print("🧠 Building RL Circuit (High Input Mode)...")
        
        self.n_input = 2
        self.n_output = 2
        
        self.input_layer = neocortex_genes.generate_cortical_layer(self.n_input, "RS")
        self.output_layer = neocortex_genes.generate_cortical_layer(self.n_output, "RS")
        
        # Init Weights (20.0 - 30.0)
        weights = np.random.uniform(20.0, 30.0, (self.n_input, self.n_output)).astype(np.float32)
        
        self.d_weights = cuda.to_device(weights)
        
        # Traces
        self.d_pre_traces = cuda.to_device(np.zeros(self.n_input, dtype=np.float32))     
        self.d_syn_traces = cuda.to_device(np.zeros((self.n_input, self.n_output), dtype=np.float32)) 
        
        self._alloc_gpu()
        
    def _alloc_gpu(self):
        def alloc_layer(n, p):
            zeros_f = np.zeros(n, dtype=np.float32)
            zeros_i = np.zeros(n, dtype=np.int32)
            return {
                'v': cuda.to_device(p['state']['v']),
                'u': cuda.to_device(p['state']['u']),
                'a': cuda.to_device(p['params']['a']),
                'b': cuda.to_device(p['params']['b']),
                'c': cuda.to_device(p['params']['c']),
                'd': cuda.to_device(p['params']['d']),
                'i': cuda.to_device(zeros_f),
                's': cuda.to_device(zeros_i)
            }
        self.gpu_in = alloc_layer(self.n_input, self.input_layer)
        self.gpu_out = alloc_layer(self.n_output, self.output_layer)

    def run_trial(self, input_idx, target_idx, current_lr, reward_baseline, temperature):
        # Reset voltages
        self.gpu_in['v'].copy_to_device(self.input_layer['state']['v'])
        self.gpu_out['v'].copy_to_device(self.output_layer['state']['v'])
        
        # Reset Synaptic Traces (Start fresh each trial)
        dims_2d = neocortex_kernels.get_dims_2d(self.n_input, self.n_output)
        neocortex_kernels.clear_trace_kernel[dims_2d[0], dims_2d[1]](self.d_syn_traces, self.n_input, self.n_output)
        
        # Note: We do NOT reset d_pre_traces here, allowing continuity (Biological)

        steps = int(SIM_TIME_PER_TRIAL / DT)
        spike_counts = np.zeros(self.n_output, dtype=np.int32)
        
        # Input (BOOSTED!)
        inp_current = np.zeros(self.n_input, dtype=np.float32)
        inp_current[input_idx] = 300.0 # ★修正: 100 -> 300
        self.gpu_in['i'].copy_to_device(inp_current)
        
        neocortex_kernels.clear_buffer_kernel[neocortex_kernels.get_dims_1d(self.n_output)](self.gpu_out['i'], self.n_output)

        dims_in = neocortex_kernels.get_dims_1d(self.n_input)
        dims_out = neocortex_kernels.get_dims_1d(self.n_output)

        for t in range(steps):
            # 1. Update Input & Pre-Trace
            neocortex_kernels.update_neuron_kernel[dims_in[0], dims_in[1]](
                self.gpu_in['v'], self.gpu_in['u'], self.gpu_in['a'], self.gpu_in['b'],
                self.gpu_in['c'], self.gpu_in['d'], self.gpu_in['i'], self.gpu_in['s'],
                DT, self.n_input
            )
            # Update Pre-Trace
            neocortex_kernels.update_pre_trace_kernel[dims_in[0], dims_in[1]](
                self.gpu_in['s'], self.d_pre_traces, TRACE_DECAY, TRACE_INCR, self.n_input
            )
            cuda.synchronize()
            
            # 2. Transmit
            neocortex_kernels.clear_buffer_kernel[dims_out[0], dims_out[1]](self.gpu_out['i'], self.n_output)
            neocortex_kernels.synapse_kernel[dims_2d[0], dims_2d[1]](
                self.gpu_in['s'], self.gpu_out['i'], self.d_weights, self.n_input, self.n_output
            )
            cuda.synchronize()
            
            # 3. Update Output
            neocortex_kernels.update_neuron_kernel[dims_out[0], dims_out[1]](
                self.gpu_out['v'], self.gpu_out['u'], self.gpu_out['a'], self.gpu_out['b'],
                self.gpu_out['c'], self.gpu_out['d'], self.gpu_out['i'], self.gpu_out['s'],
                DT, self.n_output
            )
            cuda.synchronize()
            
            spikes = self.gpu_out['s'].copy_to_host()
            spike_counts += spikes
            
            # 4. Update Synaptic Trace
            neocortex_kernels.update_synaptic_trace_kernel[dims_2d[0], dims_2d[1]](
                self.gpu_out['s'], self.d_pre_traces, self.d_syn_traces, 
                self.n_input, self.n_output
            )
            cuda.synchronize()

        # --- Decision ---
        scores = spike_counts.astype(np.float32)
        if temperature < 1e-6: temperature = 1e-6
        exp_scores = np.exp((scores - np.max(scores)) / temperature)
        probs = exp_scores / np.sum(exp_scores)
        if np.isnan(probs).any(): probs = np.ones(self.n_output) / self.n_output
        action = np.random.choice([0, 1], p=probs)

        # Reward
        raw_reward = 1.0 if action == target_idx else 0.0
        advantage = raw_reward - reward_baseline
        
        # Diagnostics
        trace_sum = np.sum(self.d_syn_traces.copy_to_host())
        w_before = self.d_weights.copy_to_host()

        # Weight Update
        neocortex_kernels.update_weight_kernel[dims_2d[0], dims_2d[1]](
            self.d_weights, self.d_syn_traces,
            advantage, current_lr,
            self.n_input, self.n_output
        )
        cuda.synchronize()
        
        w_after = self.d_weights.copy_to_host()
        delta_w = np.linalg.norm(w_after - w_before)
        
        return (action == target_idx), w_after, raw_reward, trace_sum, delta_w

def main():
    rl = RLCircuit()
    print(f"🚀 Starting Training ({TRIALS} Trials)...")
    
    history = []
    accuracy_window = []
    
    current_lr = LEARNING_RATE
    reward_baseline = 0.5 
    temp_schedule = np.linspace(TEMP_START, TEMP_END, TRIALS)
    
    for i in range(TRIALS):
        target = np.random.randint(0, 2)
        temp = temp_schedule[i]
        
        is_correct, weights, reward, trace_sum, delta_w = rl.run_trial(target, target, current_lr, reward_baseline, temp)
        
        reward_baseline = (1 - BASELINE_ALPHA) * reward_baseline + BASELINE_ALPHA * reward
        
        current_lr *= LR_DECAY
        if current_lr < MIN_LR: current_lr = MIN_LR
        
        accuracy_window.append(1 if is_correct else 0)
        if len(accuracy_window) > 20: accuracy_window.pop(0)
        acc = sum(accuracy_window) / len(accuracy_window)
        history.append(acc)
        
        if i % 20 == 0:
            print(f"Trial {i:3d}: In={target}/Tgt={target} | Acc={acc:.2f} | LR={current_lr:.3f} | TraceSum={trace_sum:.2f} | ΔW={delta_w:.4f}")
            
        if acc >= 0.95 and i > 100:
            print(f"\n✨ Solved at Trial {i}! Stable convergence.")
            break

    print("\n📊 Training Complete!")
    plt.plot(history)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Chance')
    plt.title("Reinforcement Learning Curve (Trace Logic)")
    plt.xlabel("Trial")
    plt.ylabel("Accuracy (Moving Avg)")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    if history[-1] > 0.8:
        print("\n🏆 Success! Robust learning achieved.")
    else:
        print("\n❌ Still unstable.")

if __name__ == "__main__":
    main()