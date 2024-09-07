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
template<typename T, std::size_t N>
NagatoArray<T, N>::NagatoArray(T value)
{
  for (std::size_t i = 0; i < TotalSize_; ++i)
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
NagatoArray<T, N...> Fill(T value)
{
  // Tが数値演算可能な型であるかどうかを判定する
  static_assert(
    NagatoArithmetic<T>,
    "T is not arithmetic"
  );

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
void Show_impl(const ArrayType &array)
{
  if constexpr (ArrayType::Dimension_ == 1)
  {
    std::cout << "[";
    for (std::size_t i = 0; i < ArrayType::Shapes_[0]; i++)
    {
      std::cout << array[i];
      if (i != ArrayType::Shapes_[0] - 1)
      {
        std::cout << ", ";
      }
    }
    std::cout << "]";
  }
  else
  {
    std::cout << "[";
    for (std::size_t i = 0; i < ArrayType::Shapes_[0]; i++)
    {
      Show_impl(array[i]);
      if (i != ArrayType::Shapes_[0] - 1)
      {
        std::cout << ", ";
      }
    }
    std::cout << "]";
  }
}

template<typename ArrayType>
void Show(const ArrayType &array)
{
  static_assert(
    array_c<ArrayType>,
    "Show function is only supported for NagatoArrayFamily"
  );

  Show_impl(array);
  std::cout << std::endl;
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
-> const typename NagatoArraySlice<T, N...>::Type
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
  const std::size_t start = index * stride;
  const std::size_t end = start + stride;

  return SliceType(
    this->data,
    start,
    end
  );
}
// -----------------------------------------------------------------------------

template<typename T, size_t... N>
auto NagatoArray<T, N...>::operator[](std::size_t index)
-> NagatoArraySlice<T, N...>::Type
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
  const std::size_t start = index * stride;
  const std::size_t end = start + stride;

  return SliceType(
    this->data,
    start,
    end
  );
}

// -----------------------------------------------------------------------------

template<typename T, size_t... N>
NagatoArrayInner<T, N...>::NagatoArrayInner(const std::unique_ptr<T[]> &data,
                                            std::size_t start,
                                            std::size_t end)
  : data(std::span<T, TotalSize_>(data.get() + start, end - start))
{
}

// -----------------------------------------------------------------------------

template<typename T, std::size_t N>
NagatoArrayInner<T, N>::NagatoArrayInner(const std::unique_ptr<T[]> &data,
                                         std::size_t start,
                                         std::size_t end)
  : data(std::span<T, TotalSize_>(data.get() + start, end - start))
{
}

// -----------------------------------------------------------------------------

template<typename T, std::size_t N>
const T &NagatoArrayInner<T, N>::operator()(std::size_t index) const
{
  return data[index];
}
// -----------------------------------------------------------------------------

template<typename ArrayType>
auto Copy(const ArrayType &array)
{
  static_assert(
    array_c<ArrayType>,
    "Copy function is only supported for NagatoArrayFamily"
  );

  typename ArrayType::CopyType copy(array);

  return copy;
}
}

#endif //NAGATOLIB_SRC_NARRAY_IMPL_HPP_
