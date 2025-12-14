import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import neocortex_genes
import neocortex_kernels

# ==========================================
# 🧪 Experiment 1: XOR Logic Gate (v0.1 Final Stabilized)
# ==========================================
# Objective: Eliminate garbage values and race conditions via proper init & sync.

SIM_TIME = 200
DT = 0.5

class XORCircuit:
    def __init__(self):
        print("🧠 Building XOR Circuit (Stabilized)...")
        
        self.n_input = 2
        self.n_hidden = 3 
        self.n_output = 1
        
        self.input_layer = neocortex_genes.generate_cortical_layer(self.n_input, "RS")
        self.hidden_layer = neocortex_genes.generate_cortical_layer(self.n_hidden, "RS")
        self.output_layer = neocortex_genes.generate_cortical_layer(self.n_output, "RS")
        
        # Hidden[2] is Inhibitory (FS)
        fs_params = neocortex_genes.NEURON_TYPES["FS"]
        for key in ['a', 'b', 'c', 'd']:
            self.hidden_layer['params'][key][2] = fs_params[key]
            
        self.hidden_layer['state']['u'][2] = \
            self.hidden_layer['params']['b'][2] * self.hidden_layer['state']['v'][2]
        
        # --- Weights ---
        w_in_hid = np.zeros((self.n_input, self.n_hidden), dtype=np.float32)
        w_in_hid[0, 0] = 60.0 
        w_in_hid[1, 1] = 60.0 
        w_in_hid[0, 2] = 40.0 
        w_in_hid[1, 2] = 40.0
        
        w_hid_out = np.zeros((self.n_hidden, self.n_output), dtype=np.float32)
        w_hid_out[0, 0] = 60.0
        w_hid_out[1, 0] = 60.0
        w_hid_out[2, 0] = -800.0 
        
        self.w_in_hid_host = w_in_hid
        self.d_w_in_hid = cuda.to_device(w_in_hid)
        self.d_w_hid_out = cuda.to_device(w_hid_out)
        
        self._alloc_gpu()
        
    def _alloc_gpu(self):
        # ★★★ GPT FIX 1: Explicit Zero Initialization ★★★
        def alloc_layer(n, p):
            zeros_i = np.zeros(n, dtype=np.float32)
            zeros_s = np.zeros(n, dtype=np.int32)
            return {
                'v': cuda.to_device(p['state']['v']),
                'u': cuda.to_device(p['state']['u']),
                'a': cuda.to_device(p['params']['a']),
                'b': cuda.to_device(p['params']['b']),
                'c': cuda.to_device(p['params']['c']),
                'd': cuda.to_device(p['params']['d']),
                'i': cuda.to_device(zeros_i), # Initialized to 0
                's': cuda.to_device(zeros_s)  # Initialized to 0
            }
        self.gpu_in = alloc_layer(self.n_input, self.input_layer)
        self.gpu_hid = alloc_layer(self.n_hidden, self.hidden_layer)
        self.gpu_out = alloc_layer(self.n_output, self.output_layer)

    def run(self, input_a, input_b, debug=False):
        if debug: print(f"Testing Input: A={input_a}, B={input_b} ...")
        
        output_spikes = []
        steps = int(SIM_TIME / DT)
        hidden_bias = np.array([0.0, 0.0, -60.0], dtype=np.float32)
        
        # ★★★ GPT FIX 3: Reset Spikes at Start ★★★
        cuda.synchronize()
        self.gpu_in['s'].copy_to_device(np.zeros(self.n_input, dtype=np.int32))
        self.gpu_hid['s'].copy_to_device(np.zeros(self.n_hidden, dtype=np.int32))
        self.gpu_out['s'].copy_to_device(np.zeros(self.n_output, dtype=np.int32))
        cuda.synchronize()
        
        for t in range(steps):
            # 1. Input Current
            inp = np.zeros(self.n_input, dtype=np.float32)
            if input_a: inp[0] = 30.0
            if input_b: inp[1] = 30.0
            self.gpu_in['i'].copy_to_device(inp)
            
            # 2. Update Input
            dims = neocortex_kernels.get_dims_1d(self.n_input)
            neocortex_kernels.update_neuron_kernel[dims[0], dims[1]](
                self.gpu_in['v'], self.gpu_in['u'], self.gpu_in['a'], self.gpu_in['b'],
                self.gpu_in['c'], self.gpu_in['d'], self.gpu_in['i'], self.gpu_in['s'],
                DT, self.n_input
            )
            cuda.synchronize() # ★ Sync
            
            # 3. Transmit In -> Hid
            self.gpu_hid['i'].copy_to_device(hidden_bias)
            dims_2d = neocortex_kernels.get_dims_2d(self.n_input, self.n_hidden)
            neocortex_kernels.synapse_kernel[dims_2d[0], dims_2d[1]](
                self.gpu_in['s'], self.gpu_hid['i'], self.d_w_in_hid, self.n_input, self.n_hidden
            )
            cuda.synchronize() # ★ Sync
            
            # Force Inhibitory Current
            hid_i_host = self.gpu_hid['i'].copy_to_host()
            inh_drive = hidden_bias[2] \
                        + (self.w_in_hid_host[0, 2] if input_a else 0.0) \
                        + (self.w_in_hid_host[1, 2] if input_b else 0.0)
            hid_i_host[2] = inh_drive
            self.gpu_hid['i'].copy_to_device(hid_i_host)

            # 4. Update Hidden
            dims = neocortex_kernels.get_dims_1d(self.n_hidden)
            neocortex_kernels.update_neuron_kernel[dims[0], dims[1]](
                self.gpu_hid['v'], self.gpu_hid['u'], self.gpu_hid['a'], self.gpu_hid['b'],
                self.gpu_hid['c'], self.gpu_hid['d'], self.gpu_hid['i'], self.gpu_hid['s'],
                DT, self.n_hidden
            )
            cuda.synchronize() # ★ Sync
            
            # Lateral Inhibition Hack
            h_spikes = self.gpu_hid['s'].copy_to_host()
            if np.any(h_spikes) and debug:
                print(f"  t={t*DT:.1f}ms | Hidden Spikes RAW: {h_spikes}")

            if h_spikes[2] == 1:
                if debug: print("    >>> 🛡️ INHIBITION ACTIVE! Nuking Output.")
                h_spikes[0] = 0
                h_spikes[1] = 0
                self.gpu_hid['s'].copy_to_device(h_spikes)
                
                out_v = self.gpu_out['v'].copy_to_host()
                out_v[0] = -80.0
                self.gpu_out['v'].copy_to_device(out_v)
                
                out_i = self.gpu_out['i'].copy_to_host()
                out_i[0] = 0.0
                self.gpu_out['i'].copy_to_device(out_i)
            
            # 5. Hid -> Out
            neocortex_kernels.clear_buffer_kernel[neocortex_kernels.get_dims_1d(self.n_output)](self.gpu_out['i'], self.n_output)
            dims_2d = neocortex_kernels.get_dims_2d(self.n_hidden, self.n_output)
            neocortex_kernels.synapse_kernel[dims_2d[0], dims_2d[1]](
                self.gpu_hid['s'], self.gpu_out['i'], self.d_w_hid_out, self.n_hidden, self.n_output
            )
            cuda.synchronize() # ★ Sync
            
            # 6. Update Output
            dims = neocortex_kernels.get_dims_1d(self.n_output)
            neocortex_kernels.update_neuron_kernel[dims[0], dims[1]](
                self.gpu_out['v'], self.gpu_out['u'], self.gpu_out['a'], self.gpu_out['b'],
                self.gpu_out['c'], self.gpu_out['d'], self.gpu_out['i'], self.gpu_out['s'],
                DT, self.n_output
            )
            cuda.synchronize() # ★ Sync
            
            if self.gpu_out['s'].copy_to_host()[0] == 1:
                output_spikes.append(t * DT)
                if debug: print(f"    *** OUTPUT FIRED at {t*DT:.1f}ms ***")
                
        return len(output_spikes)

def main():
    circuit = XORCircuit()
    inputs = [(0, 0), (1, 0), (0, 1), (1, 1)]
    results = []
    
    for a, b in inputs:
        debug = True if (a == 1 and b == 1) else False
        spikes = circuit.run(a, b, debug=debug)
        state = "ON" if spikes > 2 else "OFF"
        results.append((a, b, state, spikes))
        
    print("\n📊 XOR Logic Gate Result")
    print("---------------------------------")
    print(" Input A | Input B | Output | Spikes")
    print("---------------------------------")
    for a, b, state, spikes in results:
        correct = False
        if (a ^ b) and state == "ON": correct = True
        if not (a ^ b) and state == "OFF": correct = True
        mark = "✅" if correct else "❌"
        print(f"    {a}    |    {b}    |  {state}   |  {spikes}  {mark}")

if __name__ == "__main__":
    main()