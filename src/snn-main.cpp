#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <yaml-cpp/yaml.h>

#include "neuron_models/neuron.hpp"
#include "neuron_models/lif_neuron.hpp"

#include "network/network.hpp"
#include "network/network_builder.hpp"

#include "synapse_models/synapse.hpp"

#include "simulator/snn_simulator.hpp"

#include "network/initializer/normal_weight_initializer.hpp"

#include "recorder/weight_recorder.hpp"

#include "connections/all_to_all_conntection.hpp"

snnlib::WeightRecorder recorder;

// TODO: initialization.
void build_neurons(snnlib::NetworkBuilder& network_builder){
    std::shared_ptr<snnlib::AbstractSNNNeuron> input_neurons = 
        std::make_shared<snnlib::LIFNeuron>(200);
    std::shared_ptr<snnlib::AbstractSNNNeuron> output_neurons = 
        std::make_shared<snnlib::LIFNeuron>(400);

    network_builder.add_neuron("inputs", input_neurons);
    network_builder.add_neuron("outputs", output_neurons);
}

void create_synapse(snnlib::NetworkBuilder& network_builder){
    std::shared_ptr<snnlib::AbstractSNNSynapse> input_output_synapse = 
        std::make_shared<snnlib::CurrentBasedKernalSynapse>(network_builder.get_neuron("inputs"), 
            network_builder.get_neuron("outputs"), "single_exponential",
            0.1, 0, 0, 0);
    network_builder.record_synapse("syn_input_output", input_output_synapse);
}

void establish_connections(snnlib::NetworkBuilder& network_builder){
    snnlib::NormalWeightInitializer intializer;
    std::shared_ptr<snnlib::AbstractSNNConnection> connection_input_output = 
        std::make_shared<snnlib::AllToAllConnection>(network_builder.get_synapse("syn_input_output"));
    
    intializer.initialize(connection_input_output);
    recorder.record_connection_weights_to_file(std::string("data/logs/") + "syn_input_output" + std::string(".weights"), connection_input_output);
    network_builder.add_connection("conn-input-output", connection_input_output);
}

void run_simulation(snnlib::SNNNetwork* network, int time_steps, double dt){
    snnlib::SNNSimulator simulator;
    simulator.simulate(network, time_steps, dt);
}

int main(){
    YAML::Node config = YAML::LoadFile("config.yaml");

    // building network
    snnlib::NetworkBuilder network_builder;
    build_neurons(network_builder);
    create_synapse(network_builder);
    establish_connections(network_builder);
    snnlib::SNNNetwork* network = network_builder.build_network();

    // read configuration
    int time_steps = config["snn-main"]["time-steps"].as<int>();
    double dt = config["snn-main"]["dt"].as<double>();

    // run simulation
    run_simulation(network, time_steps, dt);
}