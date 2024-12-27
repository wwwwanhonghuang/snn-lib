#ifndef SNNNEURON_PARAMETERS_HPP
#define SNNNEURON_PARAMETERS_HPP
#include <memory>
#include <unordered_map>
#include <string>
#include <stdexcept>

namespace snnlib
{
    struct SNNNeuronParameters : public std::enable_shared_from_this<snnlib::SNNNeuronParameters>{
        std::unordered_map<std::string, int> parameter_index_map;
        std::unordered_map<std::string, double> parameter_value_map;

        std::shared_ptr<snnlib::SNNNeuronParameters> 
            push(const std::string& parameter_name, int parameter_id, double value){
            parameter_index_map[parameter_name] = parameter_id;
            parameter_value_map[parameter_name] = value;
            return shared_from_this();
        }

        std::shared_ptr<snnlib::SNNNeuronParameters> 
            set(const std::string& parameter_name, double value){
            parameter_value_map[parameter_name] = value;
            return shared_from_this();
        }
        double
            get(const std::string& parameter_name){
            if(parameter_value_map.find(parameter_name) == parameter_value_map.end()){
                throw std::invalid_argument(parameter_name + std::string(" not exists in parameter value map."));
            }

            return parameter_value_map[parameter_name];
        }
        
    };
} // namespace snnlib

#endif