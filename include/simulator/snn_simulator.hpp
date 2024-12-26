#ifndef SNN_SIMULATOR_HPP
#define SNN_SIMULATOR_HPP
#include <iostream>
#include "network/network.hpp"

#include "recorder/recorder.hpp"


void displayProgressBar(int progress, int total, int barWidth = 50) {
    // Calculate the fraction of completion
    double fraction = static_cast<double>(progress) / total;

    // Compute the number of '=' to display
    int completed = static_cast<int>(fraction * barWidth);

    // Construct the bar
    std::string bar(completed, '=');
    bar.resize(barWidth, ' ');

    // Display the bar
    std::cout << "\r[" << bar << "] " << static_cast<int>(fraction * 100) << "%";
    std::cout.flush();
}

namespace snnlib {
    struct SNNSimulator
    {
        private:
        public:
            SNNSimulator(){

            }
            void simulate(std::shared_ptr<snnlib::SNNNetwork> network, int time_steps, double dt, std::shared_ptr<snnlib::RecorderFacade> record_facade = nullptr){
                for(int t = 0; t < time_steps; t++){
                    displayProgressBar(t, time_steps);
                    network->evolve_states(t, dt, record_facade);
                    network->global_update();
                }
                finish_processing();
            }

            void finish_processing(){

            }
    };
}
#endif