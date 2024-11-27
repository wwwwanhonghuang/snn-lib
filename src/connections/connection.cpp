#include "connections/connection.hpp"

namespace snnlib
{
    void AbstractSNNConnection::initialize_weights(std::shared_ptr<snnlib::AbstractSNNConnectionWeightInitializer> initializer){
        initializer->initialize(std::shared_ptr<snnlib::AbstractSNNConnection>(this));   
    }
    void AbstractSNNConnection::forward_states_to_buffer(const std::vector<double>& S, double t, double* P, double dt){
        synapses->forward_states_to_buffer(weights, S, t, P, dt);
    }
    void AbstractSNNConnection::update_states_from_buffer(){}
    
} // namespace snnlib
