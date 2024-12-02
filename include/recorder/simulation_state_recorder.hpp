#ifndef SIMULATION_STATE_RECORDER_HPP
#define SIMULATION_STATE_RECORDER_HPP
#include <string>
#include <vector>
#include <ostream>

namespace snnlib
{
    struct SimulationStateRecorder
    {
        void record_input_current_to_file(const std::string& file_path, const std::vector<double>& input_current);
        private:
        void _record_input_current(std::ostream& file_path, const std::vector<double>& input_current);
    };   
}


#endif