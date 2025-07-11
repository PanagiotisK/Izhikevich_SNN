# Izhikevich Spiking Neural Network

This repository provides a Python-based implementation of a multi-layer Spiking Neural Network (SNN) using the biologically plausible Izhikevich neuron model. The project is intended for researchers, students, and enthusiasts exploring neural dynamics, spiking neuron behaviors, and computational neuroscience.

---

## Key Features

* **Izhikevich Neuron Model:**

  * Captures a wide range of realistic neuronal firing patterns using computationally efficient equations.
  * Neuron parameters \$(a,b,c,d)\$ are easily customizable, enabling simulation of various excitatory and inhibitory neuron behaviors.

* **Multi-Layer Network Architecture:**

  * Three-layer feedforward structure (1000 → 200 → 10 neurons).
  * Customizable connectivity and synaptic weight initialization to study different network dynamics.

* **Simulation Capabilities:**

  * Flexible experimentation with excitatory/inhibitory neuron balance.
  * Detailed recording of spike timings and membrane potentials.

---

## Visualization Tools

* **Spike Raster Plots:** Visualize neuronal firing activity across each layer, providing insights into neural information processing.
* **Membrane Potential Traces:** Observe detailed neuronal dynamics, including spike thresholds and reset behaviors, confirming theoretical expectations.

---

## Execution

### Requirements

Ensure you have Python installed with the following libraries:

```bash
pip install numpy matplotlib
```

### Running the Simulation

Execute the provided Python script from your terminal:

```bash
python izhikevich_spiking_neural_network.py
```

Results, including spike raster plots and membrane potential traces, will be generated upon simulation completion.

---

## Project Structure

* `izhikevich_spiking_neural_network.py`: Main script for running the simulation.
* `plots/`: Directory where generated visualizations (spike raster plots, membrane traces) are saved.

---

## Key Takeaways

* The Izhikevich neuron model offers computational efficiency combined with biological realism.
* Layered spiking neural networks enable detailed exploration of how excitatory and inhibitory neuron interactions shape neural dynamics.
* Spike raster plots and membrane potential traces are essential for analyzing and validating network behaviors.

---

## Further Information

Refer to the included documentation and comments within the Python script for detailed parameter descriptions, algorithm explanations, and insights into the theoretical background of spiking neural networks.

---
