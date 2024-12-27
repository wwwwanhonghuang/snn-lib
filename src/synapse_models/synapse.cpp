#include <iostream>
#include "synapse_models/synapse.hpp"

namespace snnlib
{
    int AbstractSNNSynapse::n_presynapse_neurons(){
        if(presynapse_neurons) return presynapse_neurons->n_neurons;
        return 0; 
    };
    int AbstractSNNSynapse::n_postsynapse_neurons(){
        if(postsynapse_neurons) return postsynapse_neurons->n_neurons;
        return 0;
    };

    void AbstractSNNSynapse::forward_states_to_buffer(
        const std::vector<double>& processed_S,      // processed presynaptic spike train
        int t,                          // current time
        double* P,                         // parameters
        double dt                          // time step
    ) {
    
    if (!P){
        std::cout << "Error: parameter P is null in synapse model." << std::endl;
        abort();
    }    
        // Iterate over presynaptic and postsynaptic neurons to update states (wS(t)) * K
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n_presynapse_neurons(); i++) {
            for (int j = 0; j < n_postsynapse_neurons(); j++) {
                int index = i * n_postsynapse_neurons() + j;  // Synapse index
                
                

                double input_value = processed_S[index]; // Scale the presynaptic spike train by the weight
                // std::cout << " - synpase input value for " << index << "-th" << " synapse = " << input_value << std::endl;
                std::vector<double> dx = synapse_dynamics(
                    input_value,  
                    &x[index * n_states_per_synapse],  // Synaptic state
                    t,
                    P,
                    dt
                );

                // std::cout << "  -- For synapse index = " << index << ": " << std::endl;

                // Update states_buffer with new state values
                for (int k = 0; k < n_states_per_synapse; k++) {
                    x_buffer[index * n_states_per_synapse + k] = x[index * n_states_per_synapse + k] + dx[k];

                    // std::cout << "      -- parameter index  " << k << " = x[" 
                    // << index * n_states_per_synapse + k << "] ("
                    // << &x[index * n_states_per_synapse + k] <<
                    // ")" << " + " << dx[k] <<
                    // "==" << x[index * n_states_per_synapse + k] + dx[k] 
                    // << " to x_buffer[" << index * n_states_per_synapse + k << "] (" << 
                    // &x_buffer[index * n_states_per_synapse + k]
                    // << ")" << std::endl;
                }
            }
        }
    }

    void AbstractSNNSynapse::_evolve_state(int weight, const std::vector<double>& S, int t, double* P, double dt){
        for(int i = 0; i < n_presynapse_neurons(); i++){
            for(int j = 0; j < n_postsynapse_neurons(); j++){
                int index = i * n_postsynapse_neurons() + j;
                std::vector<double> dx = synapse_dynamics(S[index],
                    &x[index * n_states_per_synapse], t, P, dt
                );
                for(int k = 0; k < n_states_per_synapse; k++)
                    x[index * n_states_per_synapse + k] = x[k] + dx[k];
            }
        }
    }
    void AbstractSNNSynapse::update_states_from_buffer(){
        #pragma omp parallel for
        for(int i = 0; i < n_presynapse_neurons() * n_postsynapse_neurons() * n_states_per_synapse; i++){
            x[i] = x_buffer[i];
            // std::cout << "      - connection global updation: x_buffer[i] = " << x_buffer[i] << "(" << 
            // &x_buffer[i] << ") -> " << "x[" << i << "] (" << &x[i] << ")" << std::endl;
        }
    }
}
