//
// Created by saleh on 9/17/23.
//

#pragma once

#include <iostream>

#include "CTensor.h"
#include "CRandFiller.h"

constexpr float MAX_ERR_FLOAT = 0.00000001f;

void PrintHeader() {
    std::cout << "Example Name: " << TARGETNAME << std::endl;
    std::cout << "Compiled Kernel Version: " << TARGETKERNEL << std::endl;
}
