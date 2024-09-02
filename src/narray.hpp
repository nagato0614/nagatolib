//
// Created by toru on 2024/08/31.
//

#ifndef NAGATOLIB_METAL_NARRAY_HPP_
#define NAGATOLIB_METAL_NARRAY_HPP_

#include <vector>
#include <type_traits>

// nagato::MetalLinearAlgebra
namespace nagato::mla
{

template<typename T>
constexpr bool is_metal_arithmetic_type();

template<typename T>
class NArray
{
  static_assert(
    is_mlp_arithmetic_type<T>(),
    "NArray only supports arithmetic types"
  );
 public:
 private:
  std::vector<int> shape_;
  std::vector<T> data_;
};





} // namespace nagato::mla

#include "narray_impl.hpp"

#endif //NAGATOLIB_METAL_NARRAY_HPP_
