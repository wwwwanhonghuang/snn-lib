#include "utils/simulation_utils.hpp"
#include <string>
#include <iostream>

void displayProgressBar(int progress, int total, int barWidth) {
    double fraction = static_cast<double>(progress) / total;

    int completed = static_cast<int>(fraction * barWidth);

    std::string bar(completed, '=');
    bar.resize(barWidth, ' ');

    std::cout << "\r[" << bar << "] " << static_cast<int>(fraction * 100) << "% " << progress << " / " << total;
    std::cout.flush();
}