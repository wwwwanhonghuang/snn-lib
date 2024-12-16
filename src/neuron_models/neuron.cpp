#include "neuron_models/neuron.hpp"
#include <cassert>

namespace snnlib
{
    inline void AbstractSNNNeuron::forward_states_to_buffer(int neuron_index, double I, double t, double* P, double dt){
        std::vector<double> dx = neuron_dynamics_model(neuron_index, I, &x[neuron_index * n_states], t, P, dt);
        for(int state_id = 0; state_id < n_states; state_id++){
            double new_state = x[neuron_index * n_states + state_id] + dx[state_id]; // x + dx
            x_buffer[neuron_index * n_states + state_id] = new_state;
        }
    }
    void AbstractSNNNeuron::forward_states_to_buffer(const std::vector<double>& I, double t, double* P, double dt){
        for(int neuron_index = 0; neuron_index < n_neurons; neuron_index++){
            forward_states_to_buffer(neuron_index, I[neuron_index], t, P, dt);
        }
    }

    void AbstractSNNNeuron::update_states_from_buffer(){
        std::memcpy(x.data(), x_buffer.data(), x.size() * sizeof(double));
    }

    AbstractSNNNeuron::AbstractSNNNeuron(int n_neurons, int n_states)
                    :n_neurons(n_neurons), n_states(n_states),
        x(n_neurons * n_states, 0.0),
        x_buffer(n_neurons * n_states, 0.0) {
        if (n_neurons <= 0 || n_states <= 0) {
            throw std::invalid_argument("n_neurons and n_states must be positive.");
        }
    }
    int AbstractSNNNeuron::get_n_states(){
        return n_states;
    }

    void AbstractSNNNeuron::_evolve_state(const std::vector<double>& I, double t, double* P, double dt){
        for(int i = 0; i < n_neurons; i++){
            std::vector<double> dx = neuron_dynamics_model(i, I[i], &x[i * n_states], t, P, dt);
            for(int state_id = 0; state_id < n_states; state_id++){
                double x_i = x[i * n_states + state_id] + dx[state_id];
                x[i * n_states + state_id] = x_i;
            }
        }
    }
} // namespace snnlib
