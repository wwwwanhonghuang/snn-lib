#include "recorder/simulation_state_recorder.hpp"
#include <fstream>
#include <iostream>
namespace snnlib
{
    void SimulationStateRecorder::record_input_current_to_file(const std::string& file_path, const std::vector<double>& input_current){
        std::ofstream output_stream(file_path);
        if(!output_stream){
            std::cout << "Error: cannot open the file " << file_path << std::endl;
            return;
        }
        _record_input_current(output_stream, input_current);
    }
    
    void SimulationStateRecorder::_record_input_current(std::ostream& output_stream, const std::vector<double>& input_current){
        for(size_t i = 0; i < input_current.size(); i++){
            output_stream << input_current[i] << std::endl;
        }
    }

} // namespace snnlib
