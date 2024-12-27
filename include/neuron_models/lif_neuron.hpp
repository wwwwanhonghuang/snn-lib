#ifndef LIF_NEURON_HPP
#define LIF_NEURON_HPP
#include <vector>
#include <cassert>
#include "neuron.hpp"

#include "macros.def"
#include "neuron_models/parameters.hpp"

namespace snnlib{
    struct LIFNeuron: public AbstractSNNNeuron
    {
        DEF_DYN_SYSTEM_PARAM(0, V_rest, -65.0)
        DEF_DYN_SYSTEM_PARAM(1, V_th, -40.0)
        DEF_DYN_SYSTEM_PARAM(2, V_reset, -60.0)
        DEF_DYN_SYSTEM_PARAM(3, tau_m, 1e-2)
        DEF_DYN_SYSTEM_PARAM(4, R, 1.0)
        DEF_DYN_SYSTEM_PARAM(5, t_ref, 5e-3)
        DEF_DYN_SYSTEM_PARAM(6, V_peak, 20)

        DEF_DYN_SYSTEM_STATE(1, last_t)

        static std::shared_ptr<snnlib::SNNNeuronParameters> new_parameters(){
            std::shared_ptr<snnlib::SNNNeuronParameters> parameters = 
                std::make_shared<snnlib::SNNNeuronParameters>();
            return parameters->push("V_rest", 0, -65.0)
                ->push("V_th", 1, -40.0)
                ->push("V_reset", 2, -60.0)
                ->push("tau_m", 3, 1e-2)
                ->push("R", 4, 1.0)
                ->push("t_ref", 5, 5e-3)
                ->push("V_peak", 6, 20);
        }
        
        LIFNeuron(int n_neurons, std::shared_ptr<snnlib::SNNNeuronParameters> parameters): AbstractSNNNeuron(n_neurons, 2)
        {
            neuron_dynamics_model = &LIFNeuron::neuron_dynamics;
            this->n_neurons = n_neurons;
            for(int i = 0; i < n_neurons; i++){
                x[i * n_states + OFFSET_STATE_last_t] = -1;
            }
            
            P.assign({parameters->get("V_rest"), parameters->get("V_th"), 
                parameters->get("V_reset"), parameters->get("tau_m"), 
                parameters->get("R"), parameters->get("t_ref"), parameters->get("V_peak")});
              std::cout << "Assigned parameters to P: ";
          
        
            std::cout << std::endl;
            parameter_map["V_rest"] = 0;
            parameter_map["V_th"] = 1;
            parameter_map["V_reset"] = 2;
            parameter_map["tau_m"] = 3;
            parameter_map["R"] = 4;
            parameter_map["t_ref"] = 5;
        }

        LIFNeuron(int n_neurons, double V_rest = -65.0, double V_th = -40.0, double V_reset = -60.0, 
            double tau_m = 1e-2, double t_ref = 5e-3, double R = 1.0, double peak = 20): AbstractSNNNeuron(n_neurons, 2)
        {
            neuron_dynamics_model = &LIFNeuron::neuron_dynamics;
            this->n_neurons = n_neurons;
            for(int i = 0; i < n_neurons; i++){
                x[i * n_states + OFFSET_STATE_last_t] = -1;
            }
            
            P.assign({V_rest, V_th, V_reset, tau_m, R, t_ref});
            parameter_map["V_rest"] = 0;
            parameter_map["V_th"] = 1;
            parameter_map["V_reset"] = 2;
            parameter_map["tau_m"] = 3;
            parameter_map["R"] = 4;
            parameter_map["t_ref"] = 5;
        }

        virtual void initialize();
        
        virtual double output_V(int neuron_id, double* x, double* output_P, int t, double dt);

        static std::vector<double> neuron_dynamics(int neuron_id, double I, double* x, int t, double* P, double dt);
    };

}
#endif