//
// Created by toru on 2024/08/31.
//

#ifndef NAGATOLIB_METAL_METAL_ARRAY_HPP_
#define NAGATOLIB_METAL_METAL_ARRAY_HPP_

#include <vector>

namespace nagato
{

template<typename T>
class MetalArray
{
 public:
 private:
  std::vector<int> shape_;
  std::vector<T> data_;
};

// Metal Linear Algebra
namespace mla
{
template<typename T>
MetalArray<T> Sum(const MetalArray<T> &lhs, const MetalArray<T> &rhs)
{
  return MetalArray<T>();
}
} // namespace mla
} // namespace nagato

#endif //NAGATOLIB_METAL_METAL_ARRAY_HPP_
