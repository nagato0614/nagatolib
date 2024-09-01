//
// Created by toru on 2024/08/31.
//

#include "metal_functions.hpp"

namespace nagato::mla
{

auto &MLASingleton::GetMetalBase()
{
  return metal_base_;
}

MLASingleton::MLASingleton()
{
  metal_base_ = std::make_unique<MetalBase>("../metal_kernel/linear_algebra.metal");
}

}