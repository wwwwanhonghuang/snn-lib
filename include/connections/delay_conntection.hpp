#ifndef DELAY_CONNECTION_HPP
#define DELAY_CONNECTION_HPP
#include <memory>
#include <stdexcept>

#include "connections/connection.hpp"
#include "macros.def"

namespace snnlib{
    struct DelayConnection: public AbstractSNNConnection
    {
        std::vector<int> delay_buffer;
        int delay_steps;
        public:
            DelayConnection(std::shared_ptr<snnlib::AbstractSNNSynapse> synpases, int delay_steps): 
                AbstractSNNConnection(synpases), 
                delay_buffer(synapses->n_presynapse_neurons() * synapses->n_postsynapse_neurons() * delay_steps, 0){
                    connected.assign(synapses->n_presynapse_neurons() * synapses->n_postsynapse_neurons(), true);
                this->delay_steps = delay_steps;
                if(delay_steps < 0){
                    throw std::invalid_argument("delay_steps cannot be negative");
                }
            }

            std::vector<double> process_pre_neuron_spike_trains(const std::vector<double>& S, int t, double* P, double dt) override{
                int moded_time = t % delay_steps;
                int n_synapses = synapses->n_presynapse_neurons() * synapses->n_postsynapse_neurons();
                int record_offset =  n_synapses * moded_time;
                std::copy(
                    S.begin(), S.end(),
                    delay_buffer.begin() + record_offset
                );
                // output delayed data, which is at delay_steps before.
                if(t - delay_steps < 0){
                    return std::vector<double>(n_synapses, 0.0);
                }else{
                    int offset = n_synapses * ((t - delay_steps) % delay_steps);
                    return std::vector<double>(delay_buffer.begin() + offset, 
                        delay_buffer.begin() + offset + n_synapses);
                }
            }

    };
}
#endif