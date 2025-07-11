import numpy as np
import matplotlib.pyplot as plt
import random

# -------------------------------
# Configuration
# -------------------------------
layer_sizes = [1000, 200, 10]   # neurons per layer
n_layers = len(layer_sizes)
dt = 0.25                       # time step (ms)
T = 1000                        # total simulation time (ms)
time_steps = int(T / dt)
time = np.arange(0, T, dt)

inhibitory_ratio = 0.2          # 20% inhibitory neurons

I_ext = 5.0                     # DC input current
I_ext_noise = 0.4               # 40% noise on Input current

V_membrane = -65.0              # membrane potentials
V_spike = 30                    # spike potential threshold

excitatory_neuron_types = {
    'RS':   {'a': 0.02,  'b': 0.2,  'c': -65.0, 'd': 8.0},   # Regular Spiking
    'IB':   {'a': 0.02,  'b': 0.2,  'c': -55.0, 'd': 4.0},   # Intrinsically Bursting
    'CH':   {'a': 0.02,  'b': 0.2,  'c': -50.0, 'd': 2.0},   # Chattering
    'TC':   {'a': 0.02,  'b': 0.25, 'c': -65.0, 'd': 0.05},  # Thalamo–cortical
    'RZ':   {'a': 0.1,   'b': 0.26, 'c': -65.0, 'd': 2.0},   # Resonator
}

inhibitory_neuron_types = {
    'FS':   {'a': 0.1,   'b': 0.2,  'c': -65.0, 'd': 2.0},   # Fast Spiking
    'LTS':  {'a': 0.02,  'b': 0.25, 'c': -65.0, 'd': 2.0},   # Low-Threshold Spiking
}


# Function: assign Izhikevich parameters based on inhibitory/excitatory type
def izhikevich_params(n, inhibitory_mask, excitatory_neuron_type, inhibitory_neuron_type):
    # Excitatory (eg regular spiking) vs Inhibitory (eg fast spiking) defaults
    a = np.where(inhibitory_mask, inhibitory_neuron_types[inhibitory_neuron_type]['a'], excitatory_neuron_types[excitatory_neuron_type]['a'])
    b = np.where(inhibitory_mask, inhibitory_neuron_types[inhibitory_neuron_type]['b'], excitatory_neuron_types[excitatory_neuron_type]['b'])
    c = np.where(inhibitory_mask, inhibitory_neuron_types[inhibitory_neuron_type]['c'], excitatory_neuron_types[excitatory_neuron_type]['c'])
    d = np.where(inhibitory_mask, inhibitory_neuron_types[inhibitory_neuron_type]['d'], excitatory_neuron_types[excitatory_neuron_type]['d'])
    return a, b, c, d

# -------------------------------
# Initialization
# -------------------------------
V = [np.full((n,), V_membrane) for n in layer_sizes]            # membrane potentials
u = [None] * n_layers                                           # recovery variables
a = [None] * n_layers; b = [None] * n_layers
c = [None] * n_layers; d = [None] * n_layers
inhibitory_mask = [None] * n_layers

firings = [np.zeros((n, time_steps)) for n in layer_sizes]      # spike recording
V_record_input = np.zeros((layer_sizes[0], time_steps))         # record V for output layer
V_record_output = np.zeros((layer_sizes[-1], time_steps))       # record V for output layer

excitatory_neuron_type = np.random.choice(list(excitatory_neuron_types), 1)[0]
inhibitory_neuron_type = np.random.choice(list(inhibitory_neuron_types), 1)[0]

# Parameters per layer
for l in range(n_layers):
    n = layer_sizes[l]
    mask = np.zeros(n, dtype=bool)
    mask[:int(n * inhibitory_ratio)] = True
    np.random.shuffle(mask)
    inhibitory_mask[l] = mask

    a[l], b[l], c[l], d[l] = izhikevich_params(n, mask, excitatory_neuron_type, inhibitory_neuron_type)
    u[l] = b[l] * V[l]

# Synaptic weights: excitatory positive, inhibitory negative
weights = []
for l in range(n_layers - 1):
    n_pre = layer_sizes[l]
    n_post = layer_sizes[l+1]
    w = np.random.normal(0.5, 0.1, (n_post, n_pre))
    w[:, inhibitory_mask[l]] *= -1
    weights.append(w)

# -------------------------------
# Simulation Loop
# -------------------------------
for t in range(time_steps):
    input_current = [np.zeros(n) for n in layer_sizes]

    # External input to Layer 1 (noisy current)
    ext = np.random.normal(I_ext, I_ext*I_ext_noise, layer_sizes[0])
    input_current[0] = np.clip(ext, 0, None)

    for l in range(n_layers):
        # Detect spikes (V >= 30)
        fired = V[l] >= V_spike
        firings[l][:, t] = fired.astype(float)

        # Reset if spiked
        V[l][fired] = c[l][fired]
        u[l][fired] += d[l][fired]

        # Euler integration step
        V[l] += dt * (0.04 * V[l]**2 + 5*V[l] + 140 - u[l] + input_current[l])
        u[l] += dt * a[l] * (b[l]*V[l] - u[l])

        # Pass spikes to next layer as input current
        if l < n_layers - 1:
            I_next = weights[l] @ fired.astype(float)
            input_current[l+1] += I_next

        # Record V for input layer (first layer)
        if l == 0:
            V_record_input[:, t] = V[l]

        # Record V for output layer (last layer)
        if l == n_layers - 1:
            V_record_output[:, t] = V[l]

    # if t % 200 == 0:                      # only occasionally, to avoid thousands of figures
    #     plt.figure()
    #     plt.bar(np.arange(layer_sizes[l]), fired.astype(int))
    #     plt.ylim(0, 1.1)
    #     plt.title(f'Layer {l+1}, fired at t = {t} ({t*dt:.1f} ms)')
    #     plt.xlabel('Neuron')
    #     plt.ylabel('Fired (1=yes)')
    #     plt.show()


# -------------------------------------------
#  Raster plots of the spike trains
# -------------------------------------------
time_ms = np.arange(time_steps) * dt          # horizontal axis for the plots

for l, st in enumerate(firings):          # st has shape (n_neurons, time_steps)
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # draw one vertical bar (“|”) at each spike time for every neuron
    for n in range(st.shape[0]):
        spike_times = time_ms[st[n] == 1.0]     # times where that neuron fired
        ax.vlines(spike_times,                 # x positions
                  n + 0.6, n + 1.4)            # a short vertical tick for clarity
    
    ax.set_ylim(0.5, st.shape[0] + 0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron index')
    ax.set_title(f'Layer {l + 1} – spike raster / {excitatory_neuron_type} - {inhibitory_neuron_type}')
    ax.grid(True, axis='y', linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    plt.show()


# -------------------------------
# Plot: Voltage Traces for Output Layer
# -------------------------------
n_out = layer_sizes[-1]
fig, axes = plt.subplots(5, 2, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.flatten()

for idx in range(n_out):
    ax = axes[idx]
    ax.plot(time, V_record_output[idx], linewidth=1)
    ax.set_title(f'Neuron {idx+1}')
    ax.set_ylim([-90, 40])
    ax.set_ylabel('V (mV)')
    ax.grid(True)

# Common labels
axes[-2].set_xlabel('Time (ms)')
axes[-1].set_xlabel('Time (ms)')
fig.suptitle('Output Layer Membrane Potentials', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()



# -------------------------------
# Plot: Voltage Traces for Random Neurons in Input Layer
# -------------------------------
n_in = layer_sizes[0]
fig, axes = plt.subplots(5, 2, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.flatten()
n_in_sample = sorted(random.sample(range(1, n_in + 1), 10))

for idx in range(len(n_in_sample)):
    ax = axes[idx]
    ax.plot(time, V_record_input[idx], linewidth=1)
    ax.set_title(f'Neuron {n_in_sample[idx]+1}')
    ax.set_ylim([-90, 40])
    ax.set_ylabel('V (mV)')
    ax.grid(True)

# Common labels
axes[-2].set_xlabel('Time (ms)')
axes[-1].set_xlabel('Time (ms)')
fig.suptitle('Sample from Input Layer Membrane Potentials', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()