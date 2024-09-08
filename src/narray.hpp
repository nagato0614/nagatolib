//
// Created by toru on 2024/09/07.
//

#ifndef NAGATOLIB_SRC_NARRAY_HPP_
#define NAGATOLIB_SRC_NARRAY_HPP_
#include <iostream>
#include <span>
#include <iostream>
#include <array>
#include <numeric>
#include <cassert>

#include "narray_concepts.hpp"

namespace nagato::na
{
// -----------------------------------------------------------------------------
// 前方宣言

/**
 * NagatoArray の内部クラス.
 * データは実態を持っている
 * @tparam T
 * @tparam N
 */
template<typename T, std::size_t... N>
class NagatoArray;

/**
 * NagatoArray の部分特殊化クラス
 * 次元数が1の場合はoperator[] が T 型の値を返す
 * @tparam T
 * @tparam N
 */
template<typename T, std::size_t N>
class NagatoArray<T, N>;

/**
 * NagatoArray を参照するクラス.
 * データは実態を持っておらず参照先がなくなるとデータが消える
 * @tparam T
 * @tparam N
 */
template<typename T, std::size_t... N>
class NagatoArrayInner;

/**
 * NagatoArrayInner の部分特殊化クラス
 * 次元数が1の場合はoperator[] が T 型の値を返す
 * @tparam T
 * @tparam N
 */
template<typename T, std::size_t N>
class NagatoArrayInner<T, N>;

/**
 * サポートクラス : NagatoArray のスライスを取得する
 * @tparam T
 * @tparam First
 * @tparam Rest
 */
template<typename T, std::size_t First, std::size_t... Rest>
struct NagatoArraySlice
{
  using Type = NagatoArrayInner<T, Rest...>;
};

// -----------------------------------------------------------------------------


// NagatoArray 内部型作成マクロ
#define NAGATO_ARRAY_INNER_TYPE(SELF, TYPE, DIMENSION) \
  using Self = SELF<TYPE, DIMENSION>;    \
  using ValueType = TYPE; \
  using CopyType = SELF; \
  template<typename Y> \
  using AsType = NagatoArray<Y, DIMENSION>;


// 内部定数作成マクロ (1次元部分特殊化用)
#define NAGATO_ARRAY_CONSTANTS_ONE_DIM(DIMENSION) \
  static constexpr std::size_t Dimension_ = 1; \
  static constexpr std::array<std::size_t, Dimension_> Shapes_ = {N}; \
  static constexpr std::size_t TotalSize_ = N;    \
  static_assert(NagatoArithmetic<T>);             \
  static_assert(Dimension_ > 0);

// 内部定数作成マクロ (多次元用)
#define NAGATO_ARRAY_CONSTANTS(DIMENSION) \
  static constexpr std::size_t Dimension_ = sizeof...(DIMENSION); \
  static constexpr std::array<std::size_t, Dimension_> Shapes_ = { DIMENSION...}; \
  static constexpr std::size_t TotalSize_ = ( N * ...); \
  static_assert(NagatoArithmetic<T>); \
  static_assert(Dimension_ > 0);

// -----------------------------------------------------------------------------

template<typename T, std::size_t N>
class NagatoArray<T, N>
{
 public:
  NAGATO_ARRAY_INNER_TYPE(NagatoArray, T, N);
  NAGATO_ARRAY_CONSTANTS_ONE_DIM(N);

  NagatoArray()
  {
    std::fill(this->data.get(), this->data.get() + TotalSize_, T());
  }

  NagatoArray(std::initializer_list<T> list)
  {
    if (list.size() != TotalSize_)
    {
      throw std::invalid_argument("Size is not match!");
    }

    std::copy(list.begin(), list.end(), this->data.get());
  }

  NagatoArray(const NagatoArray<T, N> &array)
  {
    std::copy(array.data.get(), array.data.get() + TotalSize_, this->data.get());
  }

  explicit NagatoArray(const NagatoArrayInner<T, N> &array)
  {
    std::copy(array.data.begin(), array.data.end(), this->data.get());
  }

  explicit NagatoArray(T value);

  explicit NagatoArray(std::unique_ptr<T[]> &&data);

  T *begin() const
  {
    return this->data.get();
  }

  T *end() const
  {
    return this->data.get() + TotalSize_;
  }

  /**
  * 指定したインデックスの要素を取得する.
  * @tparam Args
  * @param args
  * @return
  */
  const T &operator()(std::size_t index) const
  {
    if (index >= TotalSize_)
    {
      throw std::out_of_range("Index out of range!");
    }
    return data[index];
  }

  const T &operator[](std::size_t index) const
  {
    return operator()(index);
  }

  T &operator[](std::size_t index)
  {
    if (index >= TotalSize_)
    {
      throw std::out_of_range("Index out of range!");
    }
    return this->data[index];
  }

  std::unique_ptr<T[]> data = std::make_unique<T[]>(TotalSize_);
};


// -----------------------------------------------------------------------------

template<typename T, std::size_t... N>
class NagatoArray
{
 public:
  NAGATO_ARRAY_INNER_TYPE(NagatoArray, T, N...);
  NAGATO_ARRAY_CONSTANTS(N);

  NagatoArray()
  {
    std::fill(this->data.get(), this->data.get() + TotalSize_, T());
  }

  NagatoArray(const NagatoArray<T, N...> &array)
  {
    std::copy(array.data.get(), array.data.get() + TotalSize_, this->data.get());
  }

  explicit NagatoArray(const NagatoArrayInner<T, N...> &array)
  {
    std::copy(array.data.begin(), array.data.end(), this->data.get());
  }

  explicit NagatoArray(T value);

  explicit NagatoArray(std::unique_ptr<T[]> &&data);

  T *begin() const
  {
    return this->data.get();
  }

  T *end() const
  {
    return this->data.get() + TotalSize_;
  }

  /**
   * 指定したインデックスの要素を取得する.
   * 引数の数は次元数と同じでなければならない.
   * @tparam Args
   * @param args
   * @return
   */
  template<typename... Args>
  const T &operator()(Args... args) const;

  /**
   * 指定したインデックスの要素を取得する.
   * 次元数が1減った NagatoArrayInner を返す
   * @param index
   * @return
   */
  auto operator[](std::size_t index) const
  -> const typename NagatoArraySlice<T, N...>::Type;

  auto operator[](std::size_t index)
  -> typename NagatoArraySlice<T, N...>::Type;

  std::unique_ptr<T[]> data = std::make_unique<T[]>(TotalSize_);
};


// -----------------------------------------------------------------------------

template<typename T, std::size_t... N>
class NagatoArrayInner
{

 public:
  NAGATO_ARRAY_INNER_TYPE(NagatoArrayInner, T, N...);
  NAGATO_ARRAY_CONSTANTS(N);

  explicit NagatoArrayInner(std::span<T> data)
  {
    this->data = data;
  }

  explicit NagatoArrayInner(const std::unique_ptr<T[]> &data,
                            std::size_t start,
                            std::size_t end);

  T *begin() const
  {
    return this->data.data();
  }

  T *end() const
  {
    return this->data.data() + TotalSize_;
  }

  /**
  * 指定したインデックスの要素を取得する.
  * 引数の数は次元数と同じでなければならない.
  * @tparam Args
  * @param args
  * @return
  */
  template<typename... Args>
  const T &operator()(Args... args) const;

  auto operator[](std::size_t index) const
  -> const typename NagatoArraySlice<T, N...>::Type
  {
    assert(index < Shapes_[0]);

    const int stride = std::accumulate(
      Shapes_.begin() + 1,
      Shapes_.end(),
      1,
      std::multiplies<>()
    );

    using SliceType = NagatoArraySlice<T, N...>::Type;
    const std::size_t start = index * stride;

    static_assert(stride > 0);
    static_assert(stride <= TotalSize_);

    return SliceType(
      this->data.subspan(start, stride)
    );
  }

  auto operator[](std::size_t index)
  -> typename NagatoArraySlice<T, N...>::Type
  {
    assert(index < Shapes_[0]);

    const int stride = std::accumulate(
      Shapes_.begin() + 1,
      Shapes_.end(),
      1,
      std::multiplies<>()
    );

    using SliceType = NagatoArraySlice<T, N...>::Type;
    const std::size_t start = index * stride;

    return SliceType(
      this->data.subspan(start, stride)
    );
  }

  std::span<T, TotalSize_> data;
};

// -----------------------------------------------------------------------------
template<typename T, std::size_t N>
class NagatoArrayInner<T, N>
{
 public:
  NAGATO_ARRAY_INNER_TYPE(NagatoArrayInner, T, N);
  NAGATO_ARRAY_CONSTANTS_ONE_DIM(N);

  explicit NagatoArrayInner(std::span<T> data)
    : data(data)
  {
  }

  /**
   * スライスを作成する
   * @param data
   * @param start
   * @param end
   */
  explicit NagatoArrayInner(const std::unique_ptr<T[]> &data,
                            std::size_t start,
                            std::size_t end);

  T *begin() const
  {
    return this->data.data();
  }

  T *end() const
  {
    return this->data.data() + TotalSize_;
  }

  /**
  * 指定したインデックスの要素を取得する.
  * 引数の数は次元数と同じでなければならない.
  * @tparam Args
  * @param args
  * @return
  */
  const T &operator()(std::size_t index) const;

  const T &operator[](std::size_t index) const;

  T &operator()(std::size_t index);

  T &operator[](std::size_t index);

  std::span<T, TotalSize_> data;
};

// -----------------------------------------------------------------------------

/**
 * 指定した値で NagatoArray を初期化する
 * @tparam T
 * @tparam N
 * @param value
 * @return
 */
template<typename T, std::size_t... N>
NagatoArray<T, N...> Fill(T value);

// -----------------------------------------------------------------------------

template<typename T, std::size_t N>
NagatoArray<T, N> Zeros();

// -----------------------------------------------------------------------------

template<typename ArrayType>
auto Copy(const ArrayType &array);

// -----------------------------------------------------------------------------

/**
 * NagatoArray の要素を表示する
 * @tparam ArrayType
 * @param array
 */
template<typename ArrayType>
void Show(const ArrayType &array);

// -----------------------------------------------------------------------------

/**
 * NagatoArray 型を変える
 * @tparam ConvertType
 * @tparam ArrayType
 * @param array
 * @return
 */
template<typename ConvertType, typename ArrayType>
auto AsType(const ArrayType &array)
-> typename ArrayType::template AsType<ConvertType>;

} // namespace nagato::na

#include "narray_impl.hpp"
#include "narray_functions.hpp"

#endif //NAGATOLIB_SRC_NARRAY_HPP_
