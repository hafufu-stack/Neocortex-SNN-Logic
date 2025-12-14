import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import neocortex_genes
import neocortex_kernels

# ==========================================
# 🧪 Experiment 2: Reinforcement Learning (v0.2 Boosted)
# ==========================================
# Task: Input 0 -> Output 0, Input 1 -> Output 1
# Fixes: Stronger weights, Forced exploration

# Config
SIM_TIME_PER_TRIAL = 20 
TRIALS = 200            # More trials to see learning curve
DT = 0.5
LEARNING_RATE = 50.0     # Strong learning rate
TRACE_DECAY = 0.95      

class RLCircuit:
    def __init__(self):
        print("🧠 Building RL Circuit (Boosted)...")
        
        self.n_input = 2
        self.n_output = 2
        
        self.input_layer = neocortex_genes.generate_cortical_layer(self.n_input, "RS")
        self.output_layer = neocortex_genes.generate_cortical_layer(self.n_output, "RS")
        
        # ★修正: 初期ウェイトを大幅アップ (0~10 -> 30~40)
        # これで最初から確実に発火させる
        weights = np.random.uniform(30.0, 40.0, (self.n_input, self.n_output)).astype(np.float32)
        
        traces = np.zeros((self.n_input, self.n_output), dtype=np.float32)
        
        self.d_weights = cuda.to_device(weights)
        self.d_traces = cuda.to_device(traces)
        
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

    def run_trial(self, input_idx, target_idx):
        # Reset voltages (Simple reset)
        self.gpu_in['v'].copy_to_device(self.input_layer['state']['v'])
        self.gpu_out['v'].copy_to_device(self.output_layer['state']['v'])
        
        steps = int(SIM_TIME_PER_TRIAL / DT)
        spike_counts = np.zeros(self.n_output, dtype=np.int32)
        
        # Strong Input
        inp_current = np.zeros(self.n_input, dtype=np.float32)
        inp_current[input_idx] = 200.0 # Blast the input
        self.gpu_in['i'].copy_to_device(inp_current)
        
        # Clear Output Input Buffer
        neocortex_kernels.clear_buffer_kernel[neocortex_kernels.get_dims_1d(self.n_output)](self.gpu_out['i'], self.n_output)

        for t in range(steps):
            # Update Input
            dims = neocortex_kernels.get_dims_1d(self.n_input)
            neocortex_kernels.update_neuron_kernel[dims[0], dims[1]](
                self.gpu_in['v'], self.gpu_in['u'], self.gpu_in['a'], self.gpu_in['b'],
                self.gpu_in['c'], self.gpu_in['d'], self.gpu_in['i'], self.gpu_in['s'],
                DT, self.n_input
            )
            cuda.synchronize()
            
            # Transmit
            neocortex_kernels.clear_buffer_kernel[neocortex_kernels.get_dims_1d(self.n_output)](self.gpu_out['i'], self.n_output)
            dims_2d = neocortex_kernels.get_dims_2d(self.n_input, self.n_output)
            neocortex_kernels.synapse_kernel[dims_2d[0], dims_2d[1]](
                self.gpu_in['s'], self.gpu_out['i'], self.d_weights, self.n_input, self.n_output
            )
            cuda.synchronize()
            
            # Update Output
            dims = neocortex_kernels.get_dims_1d(self.n_output)
            neocortex_kernels.update_neuron_kernel[dims[0], dims[1]](
                self.gpu_out['v'], self.gpu_out['u'], self.gpu_out['a'], self.gpu_out['b'],
                self.gpu_out['c'], self.gpu_out['d'], self.gpu_out['i'], self.gpu_out['s'],
                DT, self.n_output
            )
            cuda.synchronize()
            
            # Count
            spikes = self.gpu_out['s'].copy_to_host()
            spike_counts += spikes
            
            # Trace Update (No reward yet)
            neocortex_kernels.reward_learning_kernel[dims_2d[0], dims_2d[1]](
                self.gpu_in['s'], self.gpu_out['s'],
                self.d_weights, self.d_traces,
                0.0, 
                LEARNING_RATE, TRACE_DECAY,
                self.n_input, self.n_output
            )
            cuda.synchronize()

        # 3. Decision (With Exploration)
        if spike_counts[0] > spike_counts[1]:
            action = 0
        elif spike_counts[1] > spike_counts[0]:
            action = 1
        else:
            # ★修正: 引き分けならランダムに選ぶ（探検）
            action = np.random.randint(0, 2)

        # 4. Reward
        reward = 0.0
        is_correct = False
        
        if action == target_idx:
            reward = 1.0 # Reward!
            is_correct = True
        else:
            reward = -0.5 # Punishment
            
        # Apply Reward
        dims_2d = neocortex_kernels.get_dims_2d(self.n_input, self.n_output)
        neocortex_kernels.reward_learning_kernel[dims_2d[0], dims_2d[1]](
            self.gpu_in['s'], self.gpu_out['s'], 
            self.d_weights, self.d_traces,
            reward, # ★ Dopamine Release
            LEARNING_RATE, TRACE_DECAY,
            self.n_input, self.n_output
        )
        cuda.synchronize()
        
        return is_correct, self.d_weights.copy_to_host(), spike_counts

def main():
    rl = RLCircuit()
    TRIALS_EXTENDED = 300 # ★修正: 少し長くして安定を見る
    print(f"🚀 Starting Training ({TRIALS_EXTENDED} Trials)...")
    
    history = []
    accuracy_window = []
    
    # Dynamic Learning Rate
    current_lr = LEARNING_RATE
    
    for i in range(TRIALS_EXTENDED):
        target = np.random.randint(0, 2)
        
        # ★修正: 学習率を渡せるように run_trial を改造するか、
        # ここでは簡易的に「正解率が高いときは学習をスキップする」ロジックを入れる
        # または、カーネル内の LEARNING_RATE は定数なので、
        # "報酬の値" を調整することで実質的な学習率を変える！
        
        # Calculate current accuracy
        acc = 0.5
        if len(accuracy_window) > 0:
            acc = sum(accuracy_window) / len(accuracy_window)
        
        # Adaptive Reward: If accuracy is high, reduce plasticity (Exploration -> Exploitation)
        # Acc=0.5 -> Scale=1.0
        # Acc=1.0 -> Scale=0.1
        learning_scale = 1.0 - (acc - 0.5) * 1.8 
        if learning_scale < 0.1: learning_scale = 0.1
        
        # Update global constant? No, we can't easily.
        # Instead, we rely on the logic inside run_trial to use `LEARNING_RATE`.
        # We will hack `run_trial` to accept a scaler or modify reward magnitude.
        
        # Let's just run it as is, but STOP early if solved!
        is_correct, weights, spikes = rl.run_trial(input_idx=target, target_idx=target)
        
        accuracy_window.append(1 if is_correct else 0)
        if len(accuracy_window) > 20: accuracy_window.pop(0)
        acc = sum(accuracy_window) / len(accuracy_window)
        history.append(acc)
        
        if i % 10 == 0:
            print(f"Trial {i:3d}: Tgt={target} | Spikes={spikes} | Correct={is_correct} | Acc={acc:.2f}")
            
        # ★追加: 早期終了（Early Stopping）
        # 完全に覚えたら（Acc > 0.95 が続いたら）、実験を成功として終わらせる
        if acc >= 0.95 and i > 50:
            print(f"\n✨ Solved at Trial {i}! Stopping early to prevent overfitting.")
            break

    print("\n📊 Training Complete!")
    plt.plot(history)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Chance')
    plt.title("Reinforcement Learning Curve")
    plt.xlabel("Trial")
    plt.ylabel("Accuracy (Moving Avg)")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    if history[-1] > 0.9:
        print("\n🏆 Success! The brain learned the rule via Dopamine.")
    else:
        print("\n❌ Learning incomplete.")

if __name__ == "__main__":
    main()