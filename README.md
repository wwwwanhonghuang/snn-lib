# A C++ Library for Simulating, Training and Deploying Spiking Neural Network
This library is incompleted and under developing.
A reservoir computing sample of current framework frontend usage for is as follows:

``` c++
void build_neurons(snnlib::NetworkBuilder& network_builder){
    network_builder.build_neuron<snnlib::PossionNeuron>("inputs", nullptr, "", 200, 80);
    network_builder.build_neuron<snnlib::LIFNeuron>("reservoir", nullptr, "rest_potential_initializer", 1000);
    network_builder.build_neuron<snnlib::LIFNeuron>("outputs", nullptr, "rest_potential_initializer", 16);
}

void build_synapses(snnlib::NetworkBuilder& network_builder) {
    network_builder.build_synapse<snnlib::CurrentBasedKernalSynapse>(
        "syn_input_reservoir", "inputs", "reservoir", "single_exponential", 0.1, 0, 0, 0);
    network_builder.build_synapse<snnlib::CurrentBasedKernalSynapse>(
        "syn_reservoir_output", "reservoir", "outputs", "single_exponential", 0.1, 0, 0, 0);
}

void establish_connections(snnlib::NetworkBuilder& network_builder){
    network_builder.build_connection<snnlib::AllToAllConnection>(
        "conn-input-reservoir", "syn_input_reservoir", nullptr, "connection_normal_initializer");
    network_builder.build_connection<snnlib::AllToAllConnection>(
        "conn-reservoir-output", "syn_reservoir_output", nullptr, "connection_normal_initializer");
}

void run_simulation(std::shared_ptr<snnlib::SNNNetwork> network, int time_steps, double dt,  std::shared_ptr<snnlib::RecorderFacade> recorder_facade = nullptr){
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

    // Read configuration
    int time_steps = config["snn-main"]["time-steps"].as<int>();
    double dt = config["snn-main"]["dt"].as<double>();

    std::cout << "time_steps = " << time_steps << std::endl;
    std::cout << "dt = " << dt << std::endl;

    // Build network
    snnlib::NetworkBuilder network_builder;
    build_neurons(network_builder);
    build_synapses(network_builder);
    establish_connections(network_builder);

    std::shared_ptr<snnlib::SNNNetwork> network = network_builder.build_network();

    // Build Recorders
    snnlib::ConnectionRecordCallback weight_recorder = [](const std::string& connection_name, std::shared_ptr<snnlib::AbstractSNNConnection> connection, int t, int dt) -> void{
        if(t == 0)
            snnlib::WeightRecorder::record_connection_weights_to_file(std::string("data/logs/") + connection_name + std::string(".weights"), connection);
    };

    snnlib::NeuroRecordCallback membrane_potential_recorder = [](const std::string& neuron_name, std::shared_ptr<snnlib::AbstractSNNNeuron> neuron, int t, int dt) -> void{
        snnlib::NeuronRecorder::record_membrane_potential_to_file(
            std::string("data/logs/")  + neuron_name
            + std::string("_t_") + std::to_string(t)
            + std::string(".v"), neuron
        );
    };

    std::shared_ptr<snnlib::RecorderFacade> recorder_facade = std::make_shared<snnlib::RecorderFacade>();
    recorder_facade->add_connection_record_item("conn-input-reservoir", weight_recorder);
    recorder_facade->add_connection_record_item("conn-reservoir-output", weight_recorder);

    recorder_facade->add_neuron_record_item("outputs", membrane_potential_recorder);
    recorder_facade->add_neuron_record_item("reservoir", membrane_potential_recorder);

    // Run simulation
    run_simulation(network, time_steps, dt, recorder_facade);
    config.reset();
}
```

# Building 

## Prerequisites

1. libyaml-cpp



## Build Commands

``` bash
$ cd <path-to-repository-root>
$ cmake .
$ make
# run the example program
$ ./bin/snn-main
```
The share library is generated in `<path-to-repository-root>/lib/libshared.a`



# Todo List
## Framework Interface
- [ ] Simplify the recorder building
- [ ] Simplify the connection building



## Performance
- [ ] Enable OpenMP
- [ ] CUDA Support



## Functionality
