#ifndef ALL_TO_ALL_CONNECTION_HPP
#define ALL_TO_ALL_CONNECTION_HPP
#include "connections/connection.hpp"
namespace snnlib{
    struct AllToAllConnection: public AbstractSNNConnection
    {
        public:
            AllToAllConnection(snnlib::AbstractSNNSynapse& synpases): 
                AbstractSNNConnection(synpases){}
    };
}
#endif