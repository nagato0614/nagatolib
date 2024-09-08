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
  std::transform(
    data.get(),
    data.get() + TotalSize_,
    data.get(),
    [value](T v) { return value; }
  );
}
// -----------------------------------------------------------------------------
template<typename T, std::size_t N>
NagatoArray<T, N>::NagatoArray(T value)
{
  std::transform(
    data.get(),
    data.get() + TotalSize_,
    data.get(),
    [value](T v) { return value; }
  );
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
    nagato_arithmetic_c<T>,
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
  static_assert(
    array_c<ArrayType>,
    "Show function is only supported for NagatoArrayFamily"
  );

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

template<typename T, std::size_t N>
const T &NagatoArrayInner<T, N>::operator[](std::size_t index) const
{
  return operator()(index);
}

// -----------------------------------------------------------------------------

template<typename T, std::size_t N>
T &NagatoArrayInner<T, N>::operator[](std::size_t index)
{
  if (index >= TotalSize_)
  {
    throw std::out_of_range("Index out of range!");
  }

  return this->data[index];
}

// -----------------------------------------------------------------------------

template<typename T, std::size_t N>
T &NagatoArrayInner<T, N>::operator()(std::size_t index)
{
  if (index >= TotalSize_)
  {
    throw std::out_of_range("Index out of range!");
  }
  return this->data[index];
}

// -----------------------------------------------------------------------------

template<typename T, std::size_t N>
NagatoArray<T, N> Zeros()
{
  return Fill<T, N>(static_cast<T>(0));
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

// -----------------------------------------------------------------------------

template<typename ConvertType, typename ArrayType>
auto AsType(const ArrayType &array)
-> typename ArrayType::template AsType<ConvertType>
{
  // ConvertType が算術型かどうかを判定する
  static_assert(nagato_arithmetic_c<ConvertType>);

  using ConvertArrayType = typename ArrayType::template AsType<ConvertType>;

  ConvertArrayType result;

  std::transform(
    array.begin(),
    array.end(),
    result.begin(),
    [](auto value) -> ConvertType
    {
      return static_cast<ConvertType>(value);
    }
  );

  return result;
}



}

#endif //NAGATOLIB_SRC_NARRAY_IMPL_HPP_
