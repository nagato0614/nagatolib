//
// Created by toru on 2024/08/31.
//

#ifndef NAGATOLIB_SRC_NARRAY_HPP_
#define NAGATOLIB_SRC_NARRAY_HPP_

#include <span>
#include <vector>
#include <type_traits>

#include "narray_types.hpp"
#include "narray_inner.hpp"
#include "narray_concepts.hpp"

namespace nagato::na
{

// -----------------------------------------------------------------------------

template<typename T>
class NArray
{
  using Type = T;
  using Self = NArray<T>;

  static_assert(
    NArray_concept<Self, T>,
    "NArray only supports arithmetic types"
  );
 public:

  explicit NArray(std::vector<T> &&data, std::vector<int> &&shape);

  /**
   * copy を作成する
   * @param index
   * @return
   */
  NArray_inner<T> operator[](int index) const;

  [[nodiscard]] const nVector<int> &Shape() const;
  nVector<int> shape_;
  nVector<T> data_;
};



// -----------------------------------------------------------------------------


/**
 * 初期化リストを受取り、NArrayを返す
 * @tparam T
 * @param list
 * @return
 */
template<typename T>
auto Array(std::initializer_list<T> list);
// -----------------------------------------------------------------------------

/**
 * shapeを受取り、NArrayを返す
 * @tparam T
 * @param shape
 * @return
 */
template<typename T>
NArray<T> Zeros(const std::initializer_list<int> &shape);

// -----------------------------------------------------------------------------

/**
 * 一つの値を受取り、NArrayを返す
 * @tparam T
 * @param value
 * @return
 */
template<typename T>
NArray<T> Array(T value);

// -----------------------------------------------------------------------------

/**
 * stream operator
 */
template<typename T>
std::ostream &operator<<(std::ostream &os, const NArray<T> &array);

// -----------------------------------------------------------------------------

template<typename T>
void Show(const NArray<T> &array);

} // namespace nagato::mla

#include "narray_impl.hpp"

#endif //NAGATOLIB_SRC_NARRAY_HPP_
