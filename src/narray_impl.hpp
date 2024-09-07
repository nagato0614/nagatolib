//
// Created by toru on 2024/09/07.
//

#ifndef NAGATOLIB_SRC_NARRAY_IMPL_HPP_
#define NAGATOLIB_SRC_NARRAY_IMPL_HPP_

#include <iostream>
#include <array>
#include <numeric>
#include <cassert>

#include "narray.hpp"

namespace nagato::na
{
// -----------------------------------------------------------------------------

template<typename T, size_t... N>
NagatoArray<T, N...>::NagatoArray(T value)
{
  for (std::size_t i = 0; i < TotalSize_; i++)
  {
    data[i] = value;
  }
}
// -----------------------------------------------------------------------------

template<typename T, size_t... N>
NagatoArray<T, N...>::NagatoArray(std::unique_ptr<T[]> &&data)
{
  this->data = std::move(data);
}
// -----------------------------------------------------------------------------

template<typename T, size_t... N>
NagatoArray<T, N...> Array(T value)
{

  return NagatoArray<T, N...>(value);
}

// -----------------------------------------------------------------------------
template<typename T, size_t... N>
template<typename... Args>
const T &NagatoArray<T, N...>::operator()(Args... args) const
{
  // 引数の数が次元数と同じであるかどうかを判定する
  static_assert(
    sizeof...(Args) == Dimension_,
    "Invalid index access!"
  );

  // 引数を配列に変換する
  std::array<size_t, sizeof...(Args)> index = {static_cast<size_t>(args)...};

  // インデックスを計算する
  size_t idx = 0;
  size_t stride = 1;
  for (int i = Dimension_ - 1; i >= 0; i--)
  {
    idx += index[i] * stride;
    stride *= Shapes_[i];
  }
  return data[idx];
}

// -----------------------------------------------------------------------------

template<typename T, size_t... N>
template<typename... Args>
const T &NagatoArrayInner<T, N...>::operator()(Args... args) const
{
  // 引数の数が次元数と同じであるかどうかを判定する
  static_assert(
    sizeof...(Args) == Dimension_,
    "Invalid index access!"
  );

  // 引数を配列に変換する
  std::array<size_t, sizeof...(Args)> index = {static_cast<size_t>(args)...};

  // インデックスを計算する
  size_t idx = 0;
  size_t stride = 1;
  for (int i = Dimension_ - 1; i >= 0; i--)
  {
    idx += index[i] * stride;
    stride *= Shapes_[i];
  }
  return data[idx];
}


// -----------------------------------------------------------------------------

template<typename ArrayType>
void Show(const ArrayType &array)
{
  static_assert(
    array_c<ArrayType>,
    "Show function is only supported for NagatoArrayFamily"
  );
}

// -----------------------------------------------------------------------------

template<typename T, size_t... N>
NagatoArrayInner<T, N...>::NagatoArrayInner(std::span<T> data)
{
  this->data = data;
}

// -----------------------------------------------------------------------------

template<typename T, size_t... N>
auto NagatoArray<T, N...>::operator[](std::size_t index) const
{
  // index が範囲内に収まっているかどうかを判定する
  assert(index < Shapes_[0]);

  const int stride = std::accumulate(
    Shapes_.begin() + 1,
    Shapes_.end(),
    1,
    std::multiplies<>()
  );

  using SliceType = NagatoArraySlice<T, N...>::Type;

  return SliceType(
    this->data
  );
}

// -----------------------------------------------------------------------------

template<typename T, size_t... N>
NagatoArrayInner<T, N...>::NagatoArrayInner(const std::unique_ptr<T[]> &data)
: data(std::span<T, TotalSize_>(data.get(), TotalSize_))
{
}

// -----------------------------------------------------------------------------

template<typename T, std::size_t N>
NagatoArrayInner<T, N>::NagatoArrayInner(const std::unique_ptr<T[]> &data)
: data(std::span<T, N>(data.get(), N))
{
}

// -----------------------------------------------------------------------------

template<typename T, std::size_t N>
const T &NagatoArray<T, N>::operator()(std::size_t index) const
{
  // index が shape[0] 未満であるかどうかを判定する
  static_assert(
    index < Shapes_[0],
    "Index out of range!"
  );
  return data[index];
}

// -----------------------------------------------------------------------------

template<typename T, std::size_t N>
const T &NagatoArrayInner<T, N>::operator()(std::size_t index) const
{
  return data[index];
}

}

#endif //NAGATOLIB_SRC_NARRAY_IMPL_HPP_
