import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import neocortex_genes
import neocortex_kernels

# ==========================================
# 🧪 Experiment 3: Working Memory Gating (v0.3 Visual Fix)
# ==========================================
# Objective: Control information flow using "Gating" signals.
# Structure: [Input] --(Gate In)--> [Working Memory] --(Gate Out)--> [Output]

# Config
N_NEURONS = 1000        
SIM_TIME = 600
DT = 0.5

# Parameters
W_INPUT_WM = 20.0       
W_WM_REC = 2.0          
W_WM_OUT = 20.0         

# Gating Signals
GATE_CLOSED = -50.0
GATE_OPEN = 0.0

def get_dims(n):
    tp = 256
    bg = (n + tp - 1) // tp
    return bg, tp

class GatingCircuit:
    def __init__(self):
        print("🧠 Building Working Memory Gating Circuit...")
        
        self.pop_in = neocortex_genes.generate_cortical_layer(N_NEURONS, "RS")
        self.pop_wm = neocortex_genes.generate_cortical_layer(N_NEURONS, "RS")
        self.pop_out = neocortex_genes.generate_cortical_layer(N_NEURONS, "RS")
        
        # Weights
        w_in_wm = np.eye(N_NEURONS, dtype=np.float32) * W_INPUT_WM
        np.random.seed(42)
        w_wm_rec = (np.random.rand(N_NEURONS, N_NEURONS) < 0.1).astype(np.float32) * W_WM_REC
        w_wm_out = np.eye(N_NEURONS, dtype=np.float32) * W_WM_OUT
        
        self.d_w_in_wm = cuda.to_device(w_in_wm)
        self.d_w_wm_rec = cuda.to_device(w_wm_rec)
        self.d_w_wm_out = cuda.to_device(w_wm_out)
        
        self._alloc_gpu()
        self.dim = get_dims(N_NEURONS)
        self.dim2d = neocortex_kernels.get_dims_2d(N_NEURONS, N_NEURONS)

    def _alloc_gpu(self):
        def alloc(p):
            return {
                'v': cuda.to_device(p['state']['v']),
                'u': cuda.to_device(p['state']['u']),
                'a': cuda.to_device(p['params']['a']),
                'b': cuda.to_device(p['params']['b']),
                'c': cuda.to_device(p['params']['c']),
                'd': cuda.to_device(p['params']['d']),
                'i': cuda.to_device(np.zeros(N_NEURONS, dtype=np.float32)),
                's': cuda.to_device(np.zeros(N_NEURONS, dtype=np.int32))
            }
        self.gpu_in = alloc(self.pop_in)
        self.gpu_wm = alloc(self.pop_wm)
        self.gpu_out = alloc(self.pop_out)

    def run(self):
        print("🚀 Running Gating Simulation...")
        
        # Record time points for clearer plotting
        history = {'in_t': [], 'in_id': [], 'wm_t': [], 'wm_id': [], 'out_t': [], 'out_id': []}
        gate_status = {'in': [], 'out': []}
        
        input_pattern = np.zeros(N_NEURONS, dtype=np.float32)
        input_pattern[200:400] = 50.0 
        
        steps = int(SIM_TIME / DT)
        
        for t_step in range(steps):
            time = t_step * DT
            
            # 1. Input Current
            current_in = input_pattern if time < 200 else np.zeros(N_NEURONS, dtype=np.float32)
            self.gpu_in['i'].copy_to_device(current_in)
            
            # 2. Gate Control
            bias_wm = GATE_CLOSED
            bias_out = GATE_CLOSED
            
            if 100 <= time < 200: bias_wm = GATE_OPEN   
            if 200 <= time < 400: bias_wm = 5.0         
            if 400 <= time < 500: 
                bias_wm = 5.0                           
                bias_out = GATE_OPEN                    
            
            gate_status['in'].append(1 if bias_wm >= 0 else 0)
            gate_status['out'].append(1 if bias_out >= 0 else 0)
            
            # --- Simulation Step ---
            
            # Input Layer
            neocortex_kernels.update_neuron_kernel[self.dim[0], self.dim[1]](
                self.gpu_in['v'], self.gpu_in['u'], self.gpu_in['a'], self.gpu_in['b'],
                self.gpu_in['c'], self.gpu_in['d'], self.gpu_in['i'], self.gpu_in['s'],
                DT, N_NEURONS
            )
            cuda.synchronize()
            
            # Input -> WM
            neocortex_kernels.clear_buffer_kernel[self.dim[0], self.dim[1]](self.gpu_wm['i'], N_NEURONS)
            neocortex_kernels.synapse_kernel[self.dim2d[0], self.dim2d[1]](
                self.gpu_in['s'], self.gpu_wm['i'], self.d_w_in_wm, N_NEURONS, N_NEURONS
            )
            
            # WM -> WM
            neocortex_kernels.synapse_kernel[self.dim2d[0], self.dim2d[1]](
                self.gpu_wm['s'], self.gpu_wm['i'], self.d_w_wm_rec, N_NEURONS, N_NEURONS
            )
            
            # Bias Kernel
            add_bias_kernel[self.dim[0], self.dim[1]](self.gpu_wm['i'], bias_wm, N_NEURONS)
            cuda.synchronize()
            
            # WM Update
            neocortex_kernels.update_neuron_kernel[self.dim[0], self.dim[1]](
                self.gpu_wm['v'], self.gpu_wm['u'], self.gpu_wm['a'], self.gpu_wm['b'],
                self.gpu_wm['c'], self.gpu_wm['d'], self.gpu_wm['i'], self.gpu_wm['s'],
                DT, N_NEURONS
            )
            cuda.synchronize()
            
            # WM -> Output
            neocortex_kernels.clear_buffer_kernel[self.dim[0], self.dim[1]](self.gpu_out['i'], N_NEURONS)
            neocortex_kernels.synapse_kernel[self.dim2d[0], self.dim2d[1]](
                self.gpu_wm['s'], self.gpu_out['i'], self.d_w_wm_out, N_NEURONS, N_NEURONS
            )
            
            # Output Bias & Update
            add_bias_kernel[self.dim[0], self.dim[1]](self.gpu_out['i'], bias_out, N_NEURONS)
            cuda.synchronize()
            
            neocortex_kernels.update_neuron_kernel[self.dim[0], self.dim[1]](
                self.gpu_out['v'], self.gpu_out['u'], self.gpu_out['a'], self.gpu_out['b'],
                self.gpu_out['c'], self.gpu_out['d'], self.gpu_out['i'], self.gpu_out['s'],
                DT, N_NEURONS
            )
            cuda.synchronize()
            
            # Log
            if t_step % 20 == 0:
                # Input
                ids = np.where(self.gpu_in['s'].copy_to_host() > 0)[0]
                if len(ids) > 0:
                    history['in_t'].extend([time] * len(ids))
                    history['in_id'].extend(ids)
                
                # WM
                ids = np.where(self.gpu_wm['s'].copy_to_host() > 0)[0]
                if len(ids) > 0:
                    history['wm_t'].extend([time] * len(ids))
                    history['wm_id'].extend(ids)
                
                # Output
                ids = np.where(self.gpu_out['s'].copy_to_host() > 0)[0]
                if len(ids) > 0:
                    history['out_t'].extend([time] * len(ids))
                    history['out_id'].extend(ids)

        return history, gate_status

@cuda.jit
def add_bias_kernel(buffer, bias, size):
    tid = cuda.grid(1)
    if tid < size:
        buffer[tid] += bias

def main():
    sim = GatingCircuit()
    hist, gates = sim.run()
    
    print("\n📊 Analyzing Gating Control...")
    
    # Time axis for gates
    t_gate = np.linspace(0, SIM_TIME, len(gates['in']))
    
    plt.figure(figsize=(12, 10))
    
    # 1. Input
    plt.subplot(4, 1, 1)
    plt.scatter(hist['in_t'], hist['in_id'], s=1, color='blue')
    plt.ylabel('Input')
    plt.ylim(0, N_NEURONS)
    plt.xlim(0, SIM_TIME)
    plt.title("1. Sensory Input (0-200ms)")
    plt.grid(True, alpha=0.3)
    
    # 2. Gate Signals (Overlapping visualization)
    plt.subplot(4, 1, 2)
    # Green line: Thick and semi-transparent
    plt.plot(t_gate, gates['in'], label='Input Gate (Write)', color='green', linewidth=4, alpha=0.4)
    # Red line: Thin and solid
    plt.plot(t_gate, gates['out'], label='Output Gate (Read)', color='red', linewidth=1.5)
    plt.ylabel('Gate Status')
    plt.ylim(-0.1, 1.1)
    plt.xlim(0, SIM_TIME)
    plt.yticks([0, 1]) # Show only 0 and 1
    plt.legend(loc='center right')
    plt.title("2. PFC Control Signals (Gates)")
    plt.grid(True, alpha=0.3)
    
    # 3. Working Memory
    plt.subplot(4, 1, 3)
    plt.scatter(hist['wm_t'], hist['wm_id'], s=1, color='green')
    plt.ylabel('Working Memory')
    plt.ylim(0, N_NEURONS)
    plt.xlim(0, SIM_TIME)
    plt.title("3. Working Memory Activity")
    plt.grid(True, alpha=0.3)
    
    # 4. Output
    plt.subplot(4, 1, 4)
    plt.scatter(hist['out_t'], hist['out_id'], s=1, color='red')
    plt.ylabel('Output')
    plt.ylim(0, N_NEURONS)
    plt.xlim(0, SIM_TIME)
    plt.title("4. Output (Recall at 400-500ms)")
    plt.xlabel('Time (ms)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Experiment Done.")
    print("   Check the unified X-axis (ms). The flow of information should be clear now.")

if __name__ == "__main__":
    main()