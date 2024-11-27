#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include "neuron_models/neuron.hpp"
#include "neuron_models/lif_neuron.hpp"

#include "network/network.hpp"
#include "network/network_builder.hpp"

#include "synapse_models/synapse.hpp"

#include "simulator/snn_simulator.hpp"

void build_neurons(snnlib::NetworkBuilder& network_builder){
    std::shared_ptr<snnlib::AbstractSNNNeuron> input_neurons = 
        std::make_shared<snnlib::LIFNeuron>(input_neurons);
    std::shared_ptr<snnlib::AbstractSNNNeuron> output_neurons = 
        std::make_shared<snnlib::LIFNeuron>(output_neurons);   

    network_builder.add_neuron("inputs", input_neurons.get());
    network_builder.add_neuron("outputs", output_neurons.get());
}


void  build_synapse(snnlib::NetworkBuilder& network_builder){
    std::shared_ptr<snnlib::AbstractSNNSynapse> input_output_synapse = 
        std::make_shared<snnlib::AbstractSNNSynapse>(network_builder.get_neuron("inputs"), network_builder.get_neuron("outputs"));
    network_builder.record_synapse("syn_input_output", input_output_synapse.get());
}

void connect_neurons(snnlib::NetworkBuilder& network_builder){
    std::shared_ptr<snnlib::AbstractSNNConnection> connection_input_output = 
        std::make_shared<snnlib::AbstractSNNConnection>(network_builder.get_synapse("syn_input_output"));
    
    network_builder.add_connection("conn-input-output", connection_input_output.get());
}

void simulation_main(snnlib::SNNNetwork* network, int time_steps){
    snnlib::SNNSimulator simulator;
    simulator.simulate(network, time_steps);
}

int main(){
    snnlib::NetworkBuilder network_builder;
    build_neurons(network_builder);
    build_synapse(network_builder);
    make_network_connections(network_builder);
    snnlib::SNNNetwork* network = network_builder.build_network();

    int time_step = 2000;
    simulation_main(network, time_step);
}