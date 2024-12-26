#include "connections/connection.hpp"

namespace snnlib
{
    void AbstractSNNConnection::initialize_weights(std::shared_ptr<snnlib::AbstractSNNConnectionInitializer> initializer){
        initializer->initialize(std::shared_ptr<snnlib::AbstractSNNConnection>(this));   
    }
    void AbstractSNNConnection::forward_states_to_buffer(const std::vector<double>& S, int t, double* P, double dt){
        std::vector<double> output_S = this->process_pre_neuron_spike_trains(S, t, P, dt);
        synapses->forward_states_to_buffer(output_S, t, P, dt);
    }
    void AbstractSNNConnection::update_states_from_buffer(){
        synapses->update_states_from_buffer();
    }
    
} // namespace snnlib
