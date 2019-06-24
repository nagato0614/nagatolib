//
// Created by nagato0614 on 2019-06-23.
//

#ifndef NAGATOLIB_SRC_MATH_H_
#define NAGATOLIB_SRC_MATH_H_

#include <math.h>
#include <cassert>

#include "assert.hpp"

namespace nagato::math {
// -----------------------------------------------------------------------------
// reference : https://cpplover.blogspot.com/2010/11/blog-post_20.html
template<typename Type>
constexpr Type sqrt(Type s) {
  STATIC_ASSERT_IS_ARITHMETRIC(Type);
  Type x = s / 2.0; // Is there any better way to determine initial value?
  Type last_x = 0.0; // the value one before the last step
  while (x != last_x) // until the difference is not significant
  { // apply Babylonian method step
	last_x = x;
	x = (x + s / x) / 2.0;
  }
  return x;
}
// -----------------------------------------------------------------------------

template<typename Type>
constexpr Type abs(Type s) noexcept {
  STATIC_ASSERT_IS_ARITHMETRIC(Type);
  return s < 0 ? -s : s;
}

template<typename Return, typename L, typename R>
constexpr Return max(L l, R r) noexcept {
  STATIC_ASSERT_IS_ARITHMETRIC(Return);
  STATIC_ASSERT_IS_ARITHMETRIC(L);
  STATIC_ASSERT_IS_ARITHMETRIC(R);
  return l > r ? l : r;
}

template<typename Return, typename L, typename R>
constexpr Return min(L l, R r) noexcept {
  STATIC_ASSERT_IS_ARITHMETRIC(Return);
  STATIC_ASSERT_IS_ARITHMETRIC(L);
  STATIC_ASSERT_IS_ARITHMETRIC(R);

  return l < r ? l : r;
}

template<typename Type>
constexpr bool is_nan(Type t) noexcept {
  STATIC_ASSERT_IS_ARITHMETRIC(Type);
  return !(t == t);
}

template<typename Return, typename L, typename R>
constexpr Return clamp(Return val, L low, R high) noexcept {
  STATIC_ASSERT_IS_ARITHMETRIC(Return);
  STATIC_ASSERT_IS_ARITHMETRIC(L);
  STATIC_ASSERT_IS_ARITHMETRIC(R);
  if (val < low)
	return low;
  else if (val > high)
	return high;
  else
	return val;
}

// -----------------------------------------------------------------------------

}
#endif //NAGATOLIB_SRC_MATH_H_
