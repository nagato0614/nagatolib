//
// Created by nagato0614 on 2019-06-23.
//

#ifndef NAGATOLIB_SRC_MATH_H_
#define NAGATOLIB_SRC_MATH_H_

#include <math.h>
#include <cassert>

#include "assert.hpp"

namespace nagato {

// -----------------------------------------------------------------------------
// function definition

// -----------------------------------------------------------------------------
// reference : https://cpplover.blogspot.com/2010/11/blog-post_20.html
template<typename Type>
constexpr Type sqrt(Type s) noexcept {
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
// -----------------------------------------------------------------------------


template<typename L, typename R>
constexpr auto max(L l, R r) noexcept {
	STATIC_ASSERT_IS_ARITHMETRIC(L);
	STATIC_ASSERT_IS_ARITHMETRIC(R);
	return l > r ? l : r;
}

// -----------------------------------------------------------------------------

template<typename L, typename R>
constexpr auto min(L l, R r) noexcept {
	STATIC_ASSERT_IS_ARITHMETRIC(L);
	STATIC_ASSERT_IS_ARITHMETRIC(R);

	return l < r ? l : r;
}

// -----------------------------------------------------------------------------

template<typename Type>
constexpr bool is_nan(Type t) noexcept {
	STATIC_ASSERT_IS_ARITHMETRIC(Type);
	return !(t == t);
}
// -----------------------------------------------------------------------------

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

template<typename Type>
constexpr Type calc_Pi(int iteration) noexcept {
	STATIC_ASSERT_IS_FLOATING_POINT(Type);
	Type a = 1.0;
	Type b = 1.0 / sqrt(2.0);
	Type t = 1.0 / 4.0;
	Type p = 1.0;
	Type tmp = 0.0;
	Type ret;
	for (std::size_t i = 0; i < iteration; i++) {
		tmp = a;
		a = (tmp + b) / 2.0;
		b = sqrt(tmp * b);
		t = t - (p * (a - tmp) * (a - tmp));
		p = 2.0 * p;
	}
	return (a + b) * (a + b) / (4.0 * t);
}

// -----------------------------------------------------------------------------
// Math Constant

template<typename Type = float>
constexpr Type Pi = calc_Pi<Type>(10);

// -----------------------------------------------------------------------------

}
#endif //NAGATOLIB_SRC_MATH_H_
