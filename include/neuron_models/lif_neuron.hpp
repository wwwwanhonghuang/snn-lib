#ifndef LIF_NEURON_HPP
#define LIF_NEURON_HPP
#include <vector>
#include "neuron.hpp"
namespace snnlib{
    struct LIFNeuron: public AbstractSNNNeuron
    {
        double V_rest() { return P[0]; }
        double V_th() { return P[1]; }
        double V_reset() { return P[2]; }
        double tau_m() { return P[3]; }
        double R() { return P[4]; }

        LIFNeuron(int n_neurons, double V_rest = -65.0, double V_th = -50.0, double V_reset = -70.0, 
            double tau_m = 20.0, double R = 1.0): AbstractSNNNeuron(n_neurons, 2)
        {
            neuron_dynamics_model = &LIFNeuron::neuron_dynamics;
            this->n_neurons = n_neurons;
            P.assign({V_rest, V_th, V_reset, tau_m, R});
        }
        static std::vector<double> neuron_dynamics(double I, double* x, double t, double* P, double dt){
            double V = x[0];  // Membrane potential

            // Extract parameters from P
            double V_rest = P[0];
            double V_th = P[1];
            double V_reset = P[2];
            double tau_m = P[3];
            double R = P[4];

            // Compute the membrane potential dynamics using the LIF equation
            double dV = (- (V - V_rest) + I * R) / tau_m; 

            // Check for a spike (if voltage exceeds threshold)
            if (V >= V_th) {
                x[0] = V_reset;  // Reset voltage after spike
                return {0.0};  // No change to voltage after reset
            } else {
                return {dV};  // Update membrane potential
            }
        }
        
    };
}
#endif