//
// Created by toru on 2024/08/31.
//

#ifndef NAGATOLIB_METAL_METAL_ARRAY_HPP_
#define NAGATOLIB_METAL_METAL_ARRAY_HPP_

#include <vector>
#include <type_traits>

// nagato::MetalLinearAlgebra
namespace nagato::mla
{

template<typename T>
constexpr bool is_metal_arithmetic_type()
{
  bool type_list[] =
    {
      std::is_same<T, bool>::value,
      std::is_same<T, char>::value,
      std::is_same<T, unsigned char>::value,
      std::is_same<T, short>::value,
      std::is_same<T, unsigned short>::value,
      std::is_same<T, int>::value,
      std::is_same<T, unsigned int>::value,
      std::is_same<T, long>::value,
      std::is_same<T, unsigned long>::value,
      std::is_same<T, float>::value,
#ifdef __STDCPP_BFLOAT16_T__
      std::is_same<T, half>::value,
      std::is_same<T, std::bfloat16_t>::value,
#endif
    };

  for (bool b : type_list)
  {
    if (b)
    {
      return true;
    }
  }

  return false;
}

template<typename T>
class MetalArray
{
  static_assert(
    is_mlp_arithmetic_type<T>(),
    "MetalArray only supports arithmetic types"
  );
 public:
 private:
  std::vector<int> shape_;
  std::vector<T> data_;
};

/**
 * 要素事の和を求める
 * @tparam T
 * @param lhs
 * @param rhs
 * @return
 */
template<typename T>
MetalArray<T> Sum(const MetalArray<T> &lhs, const MetalArray<T> &rhs)
{
  return MetalArray<T>();
}

} // namespace nagato::mla

#endif //NAGATOLIB_METAL_METAL_ARRAY_HPP_
