//
// Created by toru on 2024/09/07.
//

#ifndef NAGATOLIB_SRC_NARRAY_HPP_
#define NAGATOLIB_SRC_NARRAY_HPP_
#include <iostream>
#include <span>

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

template<typename T, std::size_t First, std::size_t... Rest>
struct NagatoArraySlice
{
  using Type = NagatoArrayInner<T, Rest...>;
};

// -----------------------------------------------------------------------------


template<typename T, std::size_t N>
class NagatoArray<T, N>
{
 public:

  using Self = NagatoArray<T, N>;
  using ValueType = T;

  // 次元数
  static constexpr std::size_t Dimension_ = 1;

  // 各次元のサイズ
  static constexpr std::array<std::size_t, Dimension_> Shapes_ = {N};

  // 全体のデータサイズ
  static constexpr std::size_t TotalSize_ = N;

  // concepts を使って、Tが算術型かどうかを判定する
  static_assert(NagatoArithmetic<T>);

  // N.. の個数が 1 以上かどうかを判定する
  static_assert(Dimension_ > 0);

  NagatoArray(std::initializer_list<T> list)
  {
    if (list.size() != TotalSize_)
    {
      throw std::invalid_argument("Size is not match!");
    }

    std::copy(list.begin(), list.end(), this->data.get());
  }

  explicit NagatoArray(T value);

  explicit NagatoArray(std::unique_ptr<T[]> &&data);

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

 private:
  std::unique_ptr<T[]> data = std::make_unique<T[]>(TotalSize_);
};


// -----------------------------------------------------------------------------

template<typename T, std::size_t... N>
class NagatoArray
{
 public:

  using Self = NagatoArray<T, N...>;
  using ValueType = T;

  // 次元数
  static constexpr std::size_t Dimension_ = sizeof...(N);

  // 各次元のサイズ
  static constexpr std::array<std::size_t, Dimension_> Shapes_ = {N...};

  // 全体のデータサイズ
  static constexpr std::size_t TotalSize_ = (N * ...);

  // concepts を使って、Tが算術型かどうかを判定する
  static_assert(NagatoArithmetic<T>);

  // N.. の個数が 1 以上かどうかを判定する
  static_assert(Dimension_ > 0);

  explicit NagatoArray(T value);

  explicit NagatoArray(std::unique_ptr<T[]> &&data);

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

 private:
  std::unique_ptr<T[]> data = std::make_unique<T[]>(TotalSize_);
};


// -----------------------------------------------------------------------------

template<typename T, std::size_t... N>
class NagatoArrayInner
{

 public:

  using Self = NagatoArrayInner<T, N...>;
  using ValueType = T;

  // 次元数
  static constexpr std::size_t Dimension_ = sizeof...(N);

  // 各次元のサイズ
  static constexpr std::array<std::size_t, Dimension_> Shapes_ = {N...};

  // 全体のデータサイズ
  static constexpr std::size_t TotalSize_ = (N * ...);

  // concepts を使って、Tが算術型かどうかを判定する
  static_assert(NagatoArithmetic<T>);

  // N.. の個数が 1 以上かどうかを判定する
  static_assert(Dimension_ > 0);

  explicit NagatoArrayInner(std::span<T> data);

  explicit NagatoArrayInner(const std::unique_ptr<T[]> &data,
                            std::size_t start,
                            std::size_t end);

  /**
  * 指定したインデックスの要素を取得する.
  * 引数の数は次元数と同じでなければならない.
  * @tparam Args
  * @param args
  * @return
  */
  template<typename... Args>
  const T &operator()(Args... args) const;

 private:
  std::span<T, TotalSize_> data;
};

// -----------------------------------------------------------------------------
template<typename T, std::size_t N>
class NagatoArrayInner<T, N>
{
 public:

  using Self = NagatoArrayInner<T, N>;
  using ValueType = T;

  // 次元数
  static constexpr std::size_t Dimension_ = 1;

  // 各次元のサイズ
  static constexpr std::array<std::size_t, Dimension_> Shapes_ = {N};

  // 全体のデータサイズ
  static constexpr std::size_t TotalSize_ = N;

  // concepts を使って、Tが算術型かどうかを判定する
  static_assert(NagatoArithmetic<T>);

  // N.. の個数が 1 以上かどうかを判定する
  static_assert(Dimension_ > 0);

  explicit NagatoArrayInner(std::span<T> data);

  explicit NagatoArrayInner(const std::unique_ptr<T[]> &data,
                            std::size_t start,
                            std::size_t end);

  /**
  * 指定したインデックスの要素を取得する.
  * 引数の数は次元数と同じでなければならない.
  * @tparam Args
  * @param args
  * @return
  */
  const T &operator()(std::size_t index) const;

  const T &operator[](std::size_t index) const
  {
    return operator()(index);
  }

  T &operator()(std::size_t index)
  {
    if (index >= TotalSize_)
    {
      throw std::out_of_range("Index out of range!");
    }
    return this->data[index];
  }

  T &operator[](std::size_t index)
  {
    if (index >= TotalSize_)
    {
      throw std::out_of_range("Index out of range!");
    }

    return this->data[index];
  }

 private:
  std::span<T, TotalSize_> data;
};


// -----------------------------------------------------------------------------

template<typename T, std::size_t... N>
NagatoArray<T, N...> Fill(T value);

// -----------------------------------------------------------------------------

/**
 * NagatoArray の要素を表示する
 * @tparam ArrayType
 * @param array
 */
template<typename ArrayType>
void Show(const ArrayType &array);

// -----------------------------------------------------------------------------


}

#include "narray_impl.hpp"

#endif //NAGATOLIB_SRC_NARRAY_HPP_
