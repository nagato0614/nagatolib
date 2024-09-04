//
// Created by toru on 2024/09/04.
//

#ifndef NAGATOLIB_SRC_NARRAY_INNER_IMPL_HPP_
#define NAGATOLIB_SRC_NARRAY_INNER_IMPL_HPP_

namespace nagato::na
{

template<typename T>
NArray_inner<T>::NArray_inner(SpanVector<T> &&array, nVector<T> &&shape)
{
  this->array_ = array;
  this->shape_ = shape;
}

}

#endif //NAGATOLIB_SRC_NARRAY_INNER_IMPL_HPP_
