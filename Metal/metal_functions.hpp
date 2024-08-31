//
// Created by toru on 2024/08/31.
//

#ifndef NAGATOLIB_METAL_METAL_FUNCTIONS_HPP_
#define NAGATOLIB_METAL_METAL_FUNCTIONS_HPP_
#include "metal_base.hpp"

namespace nagato::mla
{

class MLASingleton
{
 public:
  static MLASingleton &GetInstance()
  {
    static MLASingleton instance;
    return instance;
  }

  MetalBase &GetMetalBase()
  {
    return metal_base_;
  }
 private:
  MLASingleton() = default;
  ~MLASingleton() = default;

  MetalBase metal_base_;
};



}

#endif //NAGATOLIB_METAL_METAL_FUNCTIONS_HPP_
