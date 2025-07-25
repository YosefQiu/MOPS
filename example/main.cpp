#include <iostream>
#include "api/MOPS.h"
int main() {
    std::cout << "Hello from MOPS::core!" << std::endl;
    MOPS::MOPS_Init("gpu");
    return 0;
}
