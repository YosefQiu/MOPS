#include <iostream>
#include "api/MOPS.h"
int main()
{
#ifdef __INTEL_COMPILER
    std::cout << "Compiled with Intel C++ Compiler (classic)" << std::endl;
#elif defined(__INTEL_LLVM_COMPILER)
    std::cout << "Compiled with Intel oneAPI DPC++/ICX compiler (LLVM)" << std::endl;
#elif defined(__GNUC__)
    std::cout << "Compiled with GNU GCC/G++" << std::endl;
#else
    std::cout << "Compiled with unknown compiler" << std::endl;
#endif
    std::cout << "Hello from MOPS::core!" << std::endl;
    MOPS::MOPS_Init("gpu");
    return 0;
}
