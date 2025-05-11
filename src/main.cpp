#include "ggl.h"
#include "api/MOPS.h"
#include "Command.hpp"	

#include "ndarray/ndarray_group_stream.hh"

int main(int argc, char* argv[])
{
	std::optional<Command> cmd;
	try {
        cmd = Command::parse(argc, argv);
        cmd->print();
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
	
	MOPS::MOPS_Init("gpu");


	return 0;
}
