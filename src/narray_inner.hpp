//
// Created by toru on 2024/09/04.
//

#ifndef NAGATOLIB_SRC_NARRAY_INNER_HPP_
#define NAGATOLIB_SRC_NARRAY_INNER_HPP_

#include <span>

#include "narray_types.hpp"

namespace nagato::na
{
// -----------------------------------------------------------------------------
// 前方宣言

// 外部からアクセスするためのクラス
template<typename T>
class NArray;

// 添字アクセス用
template<typename T>
class NArray_inner;
// -----------------------------------------------------------------------------

template<typename T>
class NArray_inner
{
  static_assert(
    is_arithmetic_type<T>(),
    "NArray only supports arithmetic types"
  );
 public:
  explicit NArray_inner(SpanVector<T> &&array, nVector<T> &&shape);

 private:
  nVector<T> shape_;
  SpanVector<T> &array_;
};

}

#include "narray_inner_impl.hpp"


#endif //NAGATOLIB_SRC_NARRAY_INNER_HPP_
