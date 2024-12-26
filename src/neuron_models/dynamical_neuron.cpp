#include "neuron_models/dynamical_neuron.hpp"


namespace snnlib{
        
    DynamicalNeuron::DynamicalNeuron(int n_neurons, int n_states, 
        std::shared_ptr<SNNNeuronMetaStructure> meta_neuron_structure): AbstractSNNNeuron(n_neurons, n_states)
    {
        shared_this = shared_from_this();
        this->meta_neuron_structure = meta_neuron_structure;
    }
    std::shared_ptr<DynamicalNeuron> shared_this;
    std::shared_ptr<SNNNeuronMetaStructure> meta_neuron_structure;

    void DynamicalNeuron::initialize(){}
    
    double DynamicalNeuron::output_V(int neuron_id, double* x, double* output_P, int t, double dt){
        return meta_neuron_structure->output_V_callback(shared_this, neuron_id, x, output_P, t, dt);
    }

    double DynamicalNeuron::get_state(const std::string& state_name, double* x){
        if(meta_neuron_structure->state_variables.find(state_name) != meta_neuron_structure->state_variables.end()){
            int state_index = meta_neuron_structure->state_variables[state_name];
            return x[state_index];
        }else{
            assert(false);
        }
    }
    std::vector<double> DynamicalNeuron::neuron_dynamics(std::shared_ptr<snnlib::DynamicalNeuron> self, int neuron_id, double I, double* x, int t, double* P, double dt){
        return meta_neuron_structure->neuron_dynamics(shared_this, neuron_id, I, x, t, P, dt);
    }

}