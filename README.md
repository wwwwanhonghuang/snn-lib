# A C++ Library for Simulating, Training and Deploying Spiking Neural Network
This library is incompleted and under developing.
A reservoir computing sample of current framework frontend usage for is as follows:

## Code Eamples
### Example 1. Reservoir Computing Architecture with Pre-define Neuron Dynamics. 
``` c++
/* Omit header inclusions. */

void build_neurons(std::shared_ptr<snnlib::NetworkBuilder> network_builder){
    network_builder->build_neuron<snnlib::PossionNeuron>("inputs", 1, 80000);
    network_builder->build_neuron<snnlib::LIFNeuron>("reservoir", 1)->
        add_initializer("rest_potential_initializer");
    network_builder->build_neuron<snnlib::LIFNeuron>("outputs", 1)->
        add_initializer("rest_potential_initializer");
}

void build_connections(std::shared_ptr<snnlib::NetworkBuilder> network_builder) {
    std::shared_ptr<snnlib::IdenticalWeightInitializer> initializer =
        std::make_shared<snnlib::IdenticalWeightInitializer>(0.5);
    
    // rise, decay
    network_builder->build_synapse<snnlib::CurrentBasedKernalSynapse>(
        "syn_input_reservoir", "inputs", "reservoir", "double_exponential", 1e-2, 1e-2, 0.0, 0.0)
        ->build_connection<snnlib::AllToAllConnection>("conn-input-reservoir", network_builder)
        ->add_initializer(initializer);

    network_builder->build_synapse<snnlib::CurrentBasedKernalSynapse>(
        "syn_reservoir_output", "reservoir", "outputs", "double_exponential",  1e-2,  1e-2, 0.0, 0.0)
        ->build_connection<snnlib::AllToAllConnection>("conn-reservoir-output", network_builder)
        ->add_initializer(initializer);
}

void run_simulation(std::shared_ptr<snnlib::SNNNetwork> network, 
                        int time_steps, double dt, 
                        std::shared_ptr<snnlib::RecorderFacade> recorder_facade = nullptr){
    snnlib::SNNSimulator simulator;
    simulator.simulate(network, time_steps, dt, recorder_facade);
}

int main(){
    YAML::Node config;
    try {
        config = YAML::LoadFile("config.yaml");
    } catch (const std::exception& e) {
        std::cerr << "Error loading config.yaml: " << e.what() << std::endl;
        return -1;
    }

    // Read Configuration
    int time_steps = config["snn-main"]["time-steps"].as<int>();
    double dt = config["snn-main"]["dt"].as<double>();

    std::cout << "time_steps = " << time_steps << std::endl;
    std::cout << "dt = " << dt << std::endl;

    // Build Network
    std::shared_ptr<snnlib::NetworkBuilder> network_builder = 
        std::make_shared<snnlib::NetworkBuilder>();
    
    build_neurons(network_builder);
    build_connections(network_builder);

    std::shared_ptr<snnlib::SNNNetwork> network = network_builder->build_network();


    std::shared_ptr<snnlib::NeuronRecorder> neuron_recorder =
        std::make_shared<snnlib::NeuronRecorder>();
    std::shared_ptr<snnlib::ConnectionRecorder> connection_recorder =
        std::make_shared<snnlib::ConnectionRecorder>();

    // Build Recorder [optional]
    snnlib::ConnectionRecordCallback weight_recorder = 
    [](const std::string& connection_name, std::shared_ptr<snnlib::AbstractSNNConnection> connection, int t, double dt) -> void{
        if(t == 0)
            snnlib::WeightRecorder::record_connection_weights_to_file(std::string("data/logs/") + connection_name + std::string(".weights"), connection);
    };

    snnlib::ConnectionRecordCallback response_recorder = 
        [connection_recorder](const std::string& connection_name, std::shared_ptr<snnlib::AbstractSNNConnection> connection, int t, double dt) -> void{
        connection_recorder->record_synapse_response_to_file(std::string("data/logs/") + connection_name + std::string(".r"), connection, t);
    };

    snnlib::NeuroRecordCallback membrane_potential_recorder = [neuron_recorder](const std::string& neuron_name, std::shared_ptr<snnlib::AbstractSNNNeuron> neuron, int t, double dt) -> void{
        neuron_recorder->record_membrane_potential_to_file(
            std::string("data/logs/")  + neuron_name
            + std::string(".v"), neuron, t
        );
    };

    // Build Recorder
    std::shared_ptr<snnlib::RecorderFacade> recorder_facade = 
                            std::make_shared<snnlib::RecorderFacade>();
    recorder_facade->add_connection_record_item("conn-input-reservoir", weight_recorder);
    recorder_facade->add_connection_record_item("conn-reservoir-output", weight_recorder);

    recorder_facade->add_neuron_record_item("outputs", membrane_potential_recorder);
    recorder_facade->add_neuron_record_item("reservoir", membrane_potential_recorder);

    // Run Simulation
    run_simulation(network, time_steps, dt, recorder_facade);
}
```

### Example 2: Dynamically Create Neurons with Custom Dynamics and Potentials
``` C++
std::shared_ptr<data::EEGRecord> record = data::RecordLoader::load_bin_file("eeg_dataset/ethz-ieeg/ID01_1h.bin");
std::cout << "Reading eeg record from " << "eeg_dataset/ethz-ieeg/ID01_1h.bin" << std::endl;
std::cout << "samping frequency = " << record->sampling_frequency() << std::endl;
std::cout << "n_channels = " << record->n_channels() << std::endl;
std::cout << "n_samples = " << record->n_samples() << std::endl;

std::shared_ptr<snnlib::NetworkBuilder> builder = std::make_shared<snnlib::NetworkBuilder>();
std::shared_ptr<snnlib::SNNNeuronMetaStructure> meta_neuron = 
    builder->new_neuron_structure()
    ->build_output_V_callback([&record](std::shared_ptr<snnlib::DynamicalNeuron> self, 
        int neuron_id, double* x, double* output_P, int t, double dt)->double{
        double value = record->data()[t][neuron_id];
        const double threshold = 0.5;
        if(value > threshold) return 1.0;
        return 0.0;
    })
    ->build_neuron_dynamics([&record](std::shared_ptr<snnlib::DynamicalNeuron> self, int neuron_id, double I, double* x, int t, double* P, double dt)->std::vector<double>{

        int last_spiking_t = self->get_state("last_t", x);
        const double threshold = 0.5;

        /* We not maintain the membrane potential (1st element of return vector) in this neuron's dynamics,
           because this neuron designed to read data from a 2-dim vector, and spiking if the value exceed
           a threshold.
           We can simply make the 1st element be 0.0. 
           Note that, the 1st state of all kinds of neurons is the membrane potential. This is defined 
           in the snnlib::AbstractSNNNeuron class.
        */
        if(record->data()[t][neuron_id] > threshold){
            // spiking
            /* delta_last_spiking_t = t - last_spiking_t
               so that last_spiking_t can be updated to t.*/
            return {0.0, t - last_spiking_t}; 
        }
        // no spiking
        return {0.0, 0.0};
    })
    ->build_state("last_t", 1);

std::shared_ptr<snnlib::AbstractSNNNeuron> neuron = 
    builder->initialize_from(meta_neuron, 10);

return 0;
``` 

## Example 3: Possion Neuron to LIF Neuron
``` c++
#include "neuron_models/neuron.hpp"
#include "neuron_models/lif_neuron.hpp"
#include "neuron_models/possion_neuron.hpp"
#include "neuron_models/initializer.hpp"
#include "network/network.hpp"
#include "network/network_builder.hpp"
#include "network/initializer/weight_initializers.hpp"
#include "synapse_models/synapse.hpp"
#include "simulator/snn_simulator.hpp"
#include "recorder/recorder.hpp"
#include "recorder/connection_recorder.hpp"
#include "recorder/neuron_recorder.hpp"
#include "connections/all_to_all_conntection.hpp"
#include "recorder/recorder_function_examples.hpp"

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <memory>

void run_example(){
    std::shared_ptr<snnlib::NetworkBuilder> builder = std::make_shared<snnlib::NetworkBuilder>();

    // build neurons and connections.
    builder->build_neuron<snnlib::PossionNeuron>("possion_input_30hz", 10,
        snnlib::PossionNeuron::default_parameters()->set("freq", 30)
    );

    builder->build_neuron<snnlib::LIFNeuron>("lif_neuron", 10,
            snnlib::LIFNeuron::default_parameters()
                ->set("V_rest", -65.0)
                ->set("V_th", -40.0)
                ->set("V_reset", -60.0)
                ->set("tau_m",  1e-2)
                ->set("R",  1.0)
                ->set("t_ref",  5e-3)
                ->set("V_peak",  20))
    ->add_initializer("reset_potential_initializer");


    // prepare weights
    std::vector<double> weights(100, 0.0);
    // test data
    std::vector<double> single_post_neuron_connection_weight{ 
                0.35281047,  0.08003144,  0.1957476 ,  0.44817864,  0.3735116 ,
               -0.19545558,  0.19001768, -0.03027144, -0.02064377,  0.0821197
    };
    for(int pre_id = 0; pre_id < 10; pre_id++){
        for(int post_id = 0; post_id < 10; post_id++){
            weights[pre_id * 10 + post_id] = single_post_neuron_connection_weight[pre_id];
        }
    }
    
    builder->build_synapse<snnlib::CurrentBasedKernalSynapse>("double_exponential_synapse", 
        "possion_input_30hz", "lif_neuron", "double_exponential", 1e-2, 1e-2, 0, 0)
        ->build_connection<snnlib::AllToAllConnection>("full_connection_1", builder)
        ->add_initializer(std::make_shared<snnlib::SpecificWeightInitializer>(
            weights));
    
    // Build Recorder
    std::shared_ptr<snnlib::NeuronRecorder> neuron_recorder =
        std::make_shared<snnlib::NeuronRecorder>();
    std::shared_ptr<snnlib::ConnectionRecorder> connection_recorder =
        std::make_shared<snnlib::ConnectionRecorder>();

    std::shared_ptr<snnlib::RecorderFacade> recorder_facade = 
                            std::make_shared<snnlib::RecorderFacade>();
    recorder_facade->add_connection_record_item("full_connection_1", 
        snnlib::recorder_function_examples::generate_weight_recorder());

    recorder_facade->add_connection_record_item("full_connection_1", 
        snnlib::recorder_function_examples::generate_response_recorder(connection_recorder));

    recorder_facade->add_neuron_record_item("possion_input_30hz", 
        snnlib::recorder_function_examples::generate_membrane_potential_recorder(neuron_recorder));
    recorder_facade->add_neuron_record_item("lif_neuron",
        snnlib::recorder_function_examples::generate_membrane_potential_recorder(neuron_recorder));

    std::shared_ptr<snnlib::SNNNetwork> network = builder->build_network();
    snnlib::SNNSimulator simulator;
    simulator.simulate(network, 10000, 1e-4, recorder_facade);
}

int main(){
    run_example();
    return 0;
}
```

# Building 

## Prerequisites

1. libyaml-cpp
2. OpenMP


## Build Commands

``` bash
$ cd <path-to-repository-root>
$ cmake -DCMAKE_BUILD_TYPE=Release
$ make
```
The share library is generated in `<path-to-repository-root>/lib/libshared.a`



# Todo List
## Framework Interface
- [ ] Simplify the recorder building
- [ ] Simplify the connection building


## Performance
- [ ] CUDA Support

## Functionality
