//
// Created by toru on 2024/09/02.
//

#ifndef NAGATOLIB_SRC_NARRAY_IMPL_HPP_
#define NAGATOLIB_SRC_NARRAY_IMPL_HPP_

#include "narray.hpp"

#include <algorithm>
#include <numeric>
#include <iostream>
#include <vector>

namespace nagato::na
{

// -----------------------------------------------------------------------------

template<typename T>
constexpr bool is_arithmetic_type()
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

  const auto result = std::any_of(
    std::begin(type_list),
    std::end(type_list),
    [](bool x)
    {
      return x;
    }
  );

  return result;
}
// -----------------------------------------------------------------------------

template<typename T>
const nVector<int> &NArray<T>::Shape() const
{
  return this->shape_;
}
// -----------------------------------------------------------------------------

// 末端
template<typename T>
T Array_impl(T scalar)
{
  static_assert(
    is_metal_arithmetic_type<T>(),
    "NArray only supports arithmetic types"
  );
  return scalar;
}

// -----------------------------------------------------------------------------

template<typename T>
auto Array_impl(
  const std::initializer_list<T> &list
)
{
  for (const auto &x : list)
  {
    const auto result = Array_impl(x);
  }
}
// -----------------------------------------------------------------------------

template<typename T>
auto Array(const std::initializer_list<T> &list)
{
}
// -----------------------------------------------------------------------------

template<typename T>
NArray<T>::NArray(std::vector<T> &&data, std::vector<int> &&shape)
{
  this->data_ = std::move(data);
  this->shape_ = std::move(shape);
}
// -----------------------------------------------------------------------------

template<typename T>
NArray<T> Array(T value)
{
  std::vector<int> shape = {1};
  std::vector<T> data = {value};
  NArray<T> array(std::move(data), std::move(shape));
  return array;
}
// -----------------------------------------------------------------------------

template<typename T>
NArray<T> Zeros(const std::initializer_list<int> &shape)
{
  std::vector<int> shape_vec(shape);
  const int size = std::accumulate(
    shape.begin(),
    shape.end(),
    0
  );
  std::vector<T> data(size, 0);
  NArray<T> array(std::move(data), std::move(shape_vec));
  return array;
}


// -----------------------------------------------------------------------------

template<typename T>
NArray<T> NArray<T>::operator[](int index) const
{
  std::vector<int> shape = this->shape_;

  if (shape.size() == 1)
  {
    return Array(this->data_.at(index));
  }
  else
  {
    const int first_shape = shape[0];
    const int step = std::accumulate(
      shape.begin() + 1,
      shape.end(),
      0
    );
    shape.erase(shape.begin());
    std::vector<T> data;
    const int start = first_shape + (index * step);
    const int end = first_shape + ((index + 1) * step);

    // copy
    std::copy(
      this->data_.begin() + start,
      this->data_.begin() + end,
      std::back_inserter(data)
    );

    return NArray<T>(std::move(data), std::move(shape));
  }
}

// -----------------------------------------------------------------------------

template<typename T>
NArray<T> &NArray<T>::operator[](int index)
{
  if (this->shape_.size() == 1)
  {
    return this->data_.at(index);
  }

}

// -----------------------------------------------------------------------------

template<typename T>
std::ostream &operator<<(
  std::ostream &os,
  const NArray<T> &array
)
{
  os << "NArray(";
  for (const auto &x : array.Shape())
  {
    os << x << ", ";
  }
  os << ")\n";
  return os;
}


// -----------------------------------------------------------------------------

template<typename T>
void Show(const NArray<T> &array)
{
  const auto shape = array.Shape();
  if (shape.size() == 1)
  {
    std::cout << "[";
    for (const auto &x : array.data_)
    {
      std::cout << x;
      if (&x != &array.data_.back())
      {
        std::cout << ", ";
      }
    }
    std::cout << "]";
  }
  else
  {
    std::cout << "[";
    for (int i = 0; i < shape[0]; i++)
    {
      Show(array[i]);
      if (i != shape[0] - 1)
      {
        std::cout << ", ";
      }
    }
    std::cout << "]";
  }
}
}
#endif //NAGATOLIB_SRC_NARRAY_IMPL_HPP_
