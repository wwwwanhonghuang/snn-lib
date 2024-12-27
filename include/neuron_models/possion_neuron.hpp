#ifndef POSSION_NEURON_HPP
#define POSSION_NEURON_HPP
#include "macros.def"
#include "neuron_models/neuron.hpp"
#include "neuron_models/parameters.hpp"
#include <memory>
namespace snnlib{
    

    struct PossionNeuron: public AbstractSNNNeuron
    {
        static std::shared_ptr<snnlib::SNNNeuronParameters> new_parameters(){
            std::shared_ptr<snnlib::SNNNeuronParameters> parameters = 
                std::make_shared<snnlib::SNNNeuronParameters>();
            return parameters->push("freq", 0, 30)->push("t_ref", 1, 0);
        }
        
        static constexpr int _state_count = 1;

        DEF_DYN_SYSTEM_PARAM(0, freq, 30)
        DEF_DYN_SYSTEM_PARAM(1, t_ref, 30)

        DEF_DYN_SYSTEM_STATE(1, last_t)

        PossionNeuron(int n_neurons, std::shared_ptr<snnlib::SNNNeuronParameters> parameters): AbstractSNNNeuron(n_neurons, 2){
            double frequency = parameters->get("freq");
            double t_ref = parameters->get("t_ref");
            neuron_dynamics_model = &PossionNeuron::neuron_dynamics;
            this->n_neurons = n_neurons;
            for(int i = 0; i < n_neurons; i++){
                x[i * n_states + OFFSET_STATE_last_t] = -1;
            }
            P.assign({(double)frequency, t_ref});
        }

        PossionNeuron(int n_neurons, int frequency, double t_ref = 0.0): AbstractSNNNeuron(n_neurons, 2)
        {
            neuron_dynamics_model = &PossionNeuron::neuron_dynamics;
            this->n_neurons = n_neurons;
            for(int i = 0; i < n_neurons; i++){
                x[i * n_states + OFFSET_STATE_last_t] = -1;
            }
            P.assign({(double)frequency, t_ref});
        }

        virtual void initialize();
        
        virtual double output_V(int neuron_id, double* x, double* output_P, int t, double dt);

        static std::vector<double> neuron_dynamics(int neuron_id, double I, double* x, int t, double* P, double dt);
    };

}
#endif