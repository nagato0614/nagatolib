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
  static auto &GetInstance()
  {
    static MLASingleton instance;
    return instance;
  }

  auto &GetMetalBase();
 private:

  MLASingleton();
  ~MLASingleton() = default;

  std::unique_ptr<MetalBase> metal_base_;
};

}

#endif //NAGATOLIB_METAL_METAL_FUNCTIONS_HPP_
