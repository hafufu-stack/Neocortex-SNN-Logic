import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import neocortex_genes
import neocortex_kernels

# ==========================================
# 🧪 Experiment 3: Working Memory Gating (v0.3)
# ==========================================
# Objective: Control information flow using "Gating" signals.
# Structure: [Input] --(Gate In)--> [Working Memory] --(Gate Out)--> [Output]

# Config
N_NEURONS = 1000        # Size of each population
SIM_TIME = 600
DT = 0.5

# Parameters
W_INPUT_WM = 20.0       # Feedforward strength
W_WM_REC = 2.0          # Recurrent strength (Self-sustaining)
W_WM_OUT = 20.0         # Readout strength

# Gating Signals (Bias injection)
# -50.0: Gate Closed (Strong inhibition)
#   0.0: Gate Open (Disinhibition)
GATE_CLOSED = -50.0
GATE_OPEN = 0.0

def get_dims(n):
    tp = 256
    bg = (n + tp - 1) // tp
    return bg, tp

class GatingCircuit:
    def __init__(self):
        print("🧠 Building Working Memory Gating Circuit...")
        
        # 1. Populations
        self.pop_in = neocortex_genes.generate_cortical_layer(N_NEURONS, "RS")
        self.pop_wm = neocortex_genes.generate_cortical_layer(N_NEURONS, "RS")
        self.pop_out = neocortex_genes.generate_cortical_layer(N_NEURONS, "RS")
        
        # 2. Weights
        # Input -> WM (Identity mapping for demo)
        w_in_wm = np.eye(N_NEURONS, dtype=np.float32) * W_INPUT_WM
        
        # WM -> WM (Recurrent - spread to neighbors)
        # Simple random recurrent to sustain activity
        np.random.seed(42)
        w_wm_rec = (np.random.rand(N_NEURONS, N_NEURONS) < 0.1).astype(np.float32) * W_WM_REC
        
        # WM -> Output (Identity mapping)
        w_wm_out = np.eye(N_NEURONS, dtype=np.float32) * W_WM_OUT
        
        # GPU Transfer
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
        
        history = {'in': [], 'wm': [], 'out': []}
        gate_status = {'in': [], 'out': []}
        
        # Input Pattern (Active for 0-200ms)
        input_pattern = np.zeros(N_NEURONS, dtype=np.float32)
        input_pattern[200:400] = 50.0 # Stimulate neurons 200-400
        
        steps = int(SIM_TIME / DT)
        
        for t_step in range(steps):
            time = t_step * DT
            
            # --- Scenario Control ---
            # 0-100ms:   Input ON, Gate IN=CLOSED, Gate OUT=CLOSED -> WM stays silent
            # 100-200ms: Input ON, Gate IN=OPEN,   Gate OUT=CLOSED -> WM loads data
            # 200-400ms: Input OFF,Gate IN=CLOSED, Gate OUT=CLOSED -> WM holds data (Delay)
            # 400-500ms: Input OFF,Gate IN=CLOSED, Gate OUT=OPEN   -> Output fires (Recall)
            
            # 1. Input Current
            current_in = input_pattern if time < 200 else np.zeros(N_NEURONS, dtype=np.float32)
            self.gpu_in['i'].copy_to_device(current_in)
            
            # 2. Gate Control (Bias)
            bias_wm = GATE_CLOSED
            bias_out = GATE_CLOSED
            
            if 100 <= time < 200: bias_wm = GATE_OPEN   # Open Input Gate (Write)
            if 200 <= time < 400: bias_wm = 5.0         # Maintenance (Slight boost to keep alive)
            if 400 <= time < 500: 
                bias_wm = 5.0                           # Keep holding
                bias_out = GATE_OPEN                    # Open Output Gate (Read)
            
            gate_status['in'].append(1 if bias_wm >= 0 else 0)
            gate_status['out'].append(1 if bias_out >= 0 else 0)
            
            # --- Simulation Step ---
            
            # 1. Update Input Layer
            neocortex_kernels.update_neuron_kernel[self.dim[0], self.dim[1]](
                self.gpu_in['v'], self.gpu_in['u'], self.gpu_in['a'], self.gpu_in['b'],
                self.gpu_in['c'], self.gpu_in['d'], self.gpu_in['i'], self.gpu_in['s'],
                DT, N_NEURONS
            )
            cuda.synchronize()
            
            # 2. Transmit Input -> WM
            neocortex_kernels.clear_buffer_kernel[self.dim[0], self.dim[1]](self.gpu_wm['i'], N_NEURONS)
            neocortex_kernels.synapse_kernel[self.dim2d[0], self.dim2d[1]](
                self.gpu_in['s'], self.gpu_wm['i'], self.d_w_in_wm, N_NEURONS, N_NEURONS
            )
            
            # 3. Transmit WM -> WM (Recurrent)
            neocortex_kernels.synapse_kernel[self.dim2d[0], self.dim2d[1]](
                self.gpu_wm['s'], self.gpu_wm['i'], self.d_w_wm_rec, N_NEURONS, N_NEURONS
            )
            
            # 4. Update WM Layer (With Gating Bias)
            # Update kernel needs to handle bias. We add bias to 'i' buffer before update?
            # Or assume kernel takes bias. neocortex_kernels v0.2's update_neuron doesn't take bias arg.
            # Let's add bias manually to the buffer here on host? Too slow.
            # Hack: Add bias using a simple kernel or modify I_input in-place?
            # Wait, update_neuron_kernel in v0.2 takes: (v, u, a, b, c, d, I_input, spike_out, dt, total)
            # It does NOT take bias.
            # Fix: We will launch a tiny kernel to add bias to 'i' buffer before update.
            # Or just assume we can edit the kernel? No, reuse v0.2.
            # Let's add bias to the input buffer directly via a new simple kernel here.
            
            add_bias_kernel[self.dim[0], self.dim[1]](self.gpu_wm['i'], bias_wm, N_NEURONS)
            cuda.synchronize()
            
            neocortex_kernels.update_neuron_kernel[self.dim[0], self.dim[1]](
                self.gpu_wm['v'], self.gpu_wm['u'], self.gpu_wm['a'], self.gpu_wm['b'],
                self.gpu_wm['c'], self.gpu_wm['d'], self.gpu_wm['i'], self.gpu_wm['s'],
                DT, N_NEURONS
            )
            cuda.synchronize()
            
            # 5. Transmit WM -> Output
            neocortex_kernels.clear_buffer_kernel[self.dim[0], self.dim[1]](self.gpu_out['i'], N_NEURONS)
            neocortex_kernels.synapse_kernel[self.dim2d[0], self.dim2d[1]](
                self.gpu_wm['s'], self.gpu_out['i'], self.d_w_wm_out, N_NEURONS, N_NEURONS
            )
            
            # 6. Update Output Layer (With Gating Bias)
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
                history['in'].append(np.where(self.gpu_in['s'].copy_to_host() > 0)[0])
                history['wm'].append(np.where(self.gpu_wm['s'].copy_to_host() > 0)[0])
                history['out'].append(np.where(self.gpu_out['s'].copy_to_host() > 0)[0])

        return history, gate_status

# Helper Kernel for Bias
@cuda.jit
def add_bias_kernel(buffer, bias, size):
    tid = cuda.grid(1)
    if tid < size:
        buffer[tid] += bias

def main():
    sim = GatingCircuit()
    hist, gates = sim.run()
    
    print("\n📊 Analyzing Gating Control...")
    
    plt.figure(figsize=(12, 10))
    
    # 1. Input
    plt.subplot(4, 1, 1)
    for t, ids in enumerate(hist['in']):
        plt.scatter([t]*len(ids), ids, s=1, color='blue')
    plt.ylabel('Input')
    plt.xlim(0, len(hist['in']))
    plt.title("1. Sensory Input (0-200ms)")
    
    # 2. Gate Signals
    plt.subplot(4, 1, 2)
    plt.plot(gates['in'], label='Input Gate (Write)', color='green')
    plt.plot(gates['out'], label='Output Gate (Read)', color='red')
    plt.ylabel('Gate Status')
    plt.xlim(0, len(gates['in']))
    plt.legend(loc='upper right')
    plt.title("2. PFC Control Signals (Gates)")
    
    # 3. Working Memory
    plt.subplot(4, 1, 3)
    for t, ids in enumerate(hist['wm']):
        plt.scatter([t]*len(ids), ids, s=1, color='green')
    plt.ylabel('Working Memory')
    plt.xlim(0, len(hist['wm']))
    plt.title("3. Working Memory Activity")
    
    # 4. Output
    plt.subplot(4, 1, 4)
    for t, ids in enumerate(hist['out']):
        plt.scatter([t]*len(ids), ids, s=1, color='red')
    plt.ylabel('Output')
    plt.xlim(0, len(hist['out']))
    plt.title("4. Output (Recall at 400-500ms)")
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Check the timing:")
    print("   - 0-100ms: Input is ON, but Gate Closed -> WM Silent.")
    print("   - 100-200ms: Gate OPEN -> WM Loads data.")
    print("   - 200-400ms: Input OFF, Gate Closed -> WM Holds data (Reverb).")
    print("   - 400-500ms: Output Gate OPEN -> Data flows to Output.")

if __name__ == "__main__":
    main()