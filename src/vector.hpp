//
// Created by nagato0614 on 2019-06-22.
//

#ifndef NAGATOLIB_SRC_VECTOR_H_
#define NAGATOLIB_SRC_VECTOR_H_

#include <cstdlib>
#include <cassert>
#include <initializer_list>
#include <array>
#include <limits>
#include <cmath>

#include "math.hpp"

namespace nagato {
using namespace math;
// -----------------------------------------------------------------------------
// forward declaration
template<typename Primitive, std::size_t size>
class Vector;
template<class Lhv, class Operator, class Rhv>
class Expression;
class Plus;
class Minus;
class Division;
class Multi;
class InnerProduct;
class SquareRoot;
class NormExpression;
class EuclideanDistance;
class AbsoluteValue;

// -----------------------------------------------------------------------------
template<typename Type>
using remove_const_reference
= std::remove_const_t<std::remove_reference_t<Type>>;

// -----------------------------------------------------------------------------

/**
 * 固定長のvector演算クラス
 * @tparam Primitive vectorで扱う型
 * @tparam size vectorに保存するデータ数
 */
template<typename Primitive, std::size_t size>
class Vector {
  static_assert(std::is_arithmetic<Primitive>(),
				"Primitive Type is not Arithmetric!");

 public:
  using Array = std::array<Primitive, size>;
  using Size = std::size_t;
  using Own_Type = Vector<Primitive, size>;

  constexpr
  explicit Vector(Primitive p = 0.0) noexcept {
	for (auto &i : array_)
	  i = p;
  }

  constexpr Vector(const Vector<Primitive, size> &v)
  noexcept : array_(v) {}

  constexpr Vector(const Vector<Primitive, size> &&v)
  noexcept : array_(v) {}

  constexpr Vector(const std::initializer_list<Primitive> &init) noexcept {
	assert(init.size() <= size);
	for (Size i = 0; i < init.size(); i++)
	  array_[i] = *(init.begin() + i);
  }

  ~Vector() = default;

  template<class E>
  constexpr Vector(const E &expression) noexcept {
	for (std::size_t i = 0; i < size; i++)
	  array_[i] = expression[i];
  }

  template<class E>
  constexpr Vector &operator+=(const E &expression) noexcept {
	for (int i = 0; i < size; i++)
	  array_[i] += expression[i];
  }

  template<class E>
  constexpr Vector &operator-=(const E &expression) noexcept {
	for (int i = 0; i < size; i++)
	  array_[i] -= expression[i];
  }

  template<class E>
  constexpr Vector &operator*=(const E &expression) noexcept {
	for (int i = 0; i < size; i++)
	  array_[i] *= expression[i];
  }

  template<class E>
  constexpr Vector &operator/=(const E &expression) noexcept {
	for (int i = 0; i < size; i++)
	  array_[i] /= expression[i];
  }

  template<class E>
  constexpr Vector<Primitive, size> &operator=(const E &expression)
  & noexcept {
	for (Size i = 0; i < size; i++)
	  (*this)[i] = expression[i];
	return *this;
  }

  constexpr Primitive &operator[](Size index)
  & noexcept {
	assert(0 <= index && index < size);
	return array_[index];
  }

  constexpr const Primitive &operator[](Size index)
  const & noexcept {
	assert(0 <= index && index < size);
	return array_[index];
  }

  constexpr const Primitive operator[](Size index)
  const && noexcept {
	assert(0 <= index && index < size);
	return array_[index];
  }

  constexpr void Sqrt() noexcept {
	for (Size i = 0; i < size; i++)
	  array_[i] = sqrt(array_[i]);
  }

  constexpr Size GetArraySize()
  const noexcept {
	return size;
  }

  constexpr bool HasNan()
  const noexcept {
	for (const auto &i : array_)
	  if (is_nan(i))
		return false;
	return true;
  }

  constexpr Primitive Max()
  const noexcept {
	Primitive max = 0;
	for (const auto &i : array_)
	  max = max(max, i);
	return max;
  }

  constexpr Primitive Min()
  const noexcept {
	Primitive min = 0;
	for (const auto &i : array_)
	  min = min(min, i);
	return min;
  }

  template<typename Type>
  constexpr void Fill(Type t) noexcept {
	STATIC_ASSERT_IS_ARITHMETRIC(Type);
	for (auto &i : array_)
	  i = static_cast<Primitive>(t);
  }

  template<typename L, typename R>
  constexpr void Clmap(L l, R r) noexcept {
	for (auto &i : array_)
	  i = Clamp(i, l, r);
  }

  constexpr Primitive Sum()
  const noexcept {
	Primitive sum = 0.0;
	for (const auto &i : array_)
	  sum += i;
	return sum;
  }
 private:
  Array array_ = {0};
};

// ベクトル演算関連の関数群
// -----------------------------------------------------------------------------
template<class Lhv, class Rhv>
constexpr inline auto Dot(const Lhv &l, const Rhv &r) noexcept {
  return Expression<Lhv, InnerProduct, Rhv>(l, r).Eval();
}

template<class Type>
constexpr inline auto Sqrt(const Type &t) noexcept {
  return Expression<Type, SquareRoot, Type>(t, t);
}

template<class Type>
constexpr inline auto Norm(const Type &t) noexcept {
  remove_const_reference<decltype(t[0])> sum = 0.0;
  for (int i = 0; i < t.GetArraySize(); i++)
	sum += t[i] * t[i];
  return sum;
}

template<class Type>
constexpr inline auto Sum(const Type &t) noexcept {
  remove_const_reference<decltype(t[0])> sum = 0.0;
  for (int i = 0; i < t.GetArraySize(); i++)
	sum += t[i];
  return sum;
}

template<class Type>
constexpr inline auto Abs(const Type &t) noexcept {
  return Expression<Type, AbsoluteValue, Type>(t, t);
}

template<class Lhv, class Rhv>
constexpr inline auto Distance(const Lhv &l, const Rhv &r) noexcept {
  return sqrt(Norm(l - r));
}

template<class Lhv, class Rhv>
constexpr inline Expression<Lhv, Plus, Rhv> operator+(const Lhv &l,
													  const Rhv &r) noexcept {
  return Expression<Lhv, Plus, Rhv>(l, r);
}

template<class Lhv, class Rhv>
constexpr inline Expression<Lhv, Minus, Rhv> operator-(const Lhv &l,
													   const Rhv &r) noexcept {
  return Expression<Lhv, Minus, Rhv>(l, r);
}

template<class Lhv, class Rhv>
constexpr inline Expression<Lhv, Multi, Rhv> operator*(const Lhv &l,
													   const Rhv &r) noexcept {
  return Expression<Lhv, Multi, Rhv>(l, r);
}

template<class Lhv, class Rhv>
constexpr inline Expression<Lhv, Division, Rhv> operator/(const Lhv &l,
														  const Rhv &r) noexcept {
  static_assert(!std::is_arithmetic<Rhv>(), "Right Value is arithmetic");
  return Expression<Lhv, Division, Rhv>(l, r);
}


// -----------------------------------------------------------------------------

template<class Lhv, class Operator, class Rhv>
class Expression {
 public:
  constexpr Expression(const Lhv &l, const Rhv &r)
  noexcept : lhv_(l), rhv_(r) {}

  constexpr auto operator[](std::size_t index)
  const noexcept {
	return Operator::Apply(lhv_, rhv_, index);
  }

  constexpr auto Eval()
  const noexcept {
	return Operator::Apply(lhv_, rhv_, 0);
  }

  constexpr std::size_t GetArraySize()
  const noexcept {
	if constexpr (!std::is_arithmetic<Lhv>()) {
	  return lhv_.GetArraySize();
	} else {
	  return rhv_.GetArraySize();
	}
  }
 private:
  const Lhv &lhv_;
  const Rhv &rhv_;
};

// -----------------------------------------------------------------------------

class AbsoluteValue {
 public:
  template<class Lhv, class Rhv>
  constexpr static inline auto Apply(const Lhv &l,
									 const Rhv &r,
									 std::size_t index) noexcept {
	return static_cast<remove_const_reference<
		decltype(l[index])>>(abs(l[index]));
  }
};

// -----------------------------------------------------------------------------

class EuclideanDistance {
 public:
  template<class Lhv, class Rhv>
  constexpr static inline auto Apply(const Lhv &l,
									 const Rhv &r,
									 std::size_t index) noexcept {
	return l[index] * l[index];
  }
};

// -----------------------------------------------------------------------------

class SquareRoot {
 public:
  template<class Lhv, class Rhv>
  constexpr static inline auto Apply(const Lhv &l,
									 const Rhv &r,
									 std::size_t index) noexcept {
	return static_cast<remove_const_reference<
		decltype(l[index])>>(sqrt(l[index]));
  }
};

// -----------------------------------------------------------------------------

class NormExpression {
 public:
  template<class Lhv, class Rhv>
  constexpr static inline auto Apply(const Lhv &l,
									 const Rhv &r,
									 std::size_t index) noexcept {
	return l[index] * l[index];
  }
};

// -----------------------------------------------------------------------------

class InnerProduct {
 public:
  template<class Lhv, class Rhv>
  constexpr static inline auto Apply(const Lhv &l,
									 const Rhv &r,
									 std::size_t index) noexcept {
	std::size_t size = 0;
	if constexpr (!std::is_arithmetic<Lhv>()) {
	  size = l.GetArraySize();
	} else {
	  size = r.GetArraySize();
	}

	remove_const_reference<decltype(l[0])> sum = 0;
	for (std::size_t i = 0; i < size; i++)
	  sum += (l[i] * r[i]);
	return sum;
  }
};

// -----------------------------------------------------------------------------


class Plus {
 public:
  template<class Lhv, class Rhv>
  constexpr static inline auto Apply(const Lhv &l,
									 const Rhv &r,
									 std::size_t index) noexcept {
	if constexpr (std::is_arithmetic<Lhv>() &&
		std::is_arithmetic<Rhv>()) {
	  return l + r;
	} else if constexpr (std::is_arithmetic<Lhv>() &&
		!std::is_arithmetic<Rhv>()) {
	  return l + r[index];
	} else if constexpr(!std::is_arithmetic<Lhv>() &&
		std::is_arithmetic<Rhv>()) {
	  return l[index] + r;
	} else {
	  return l[index] + r[index];
	}
  }
};

// -----------------------------------------------------------------------------


class Multi {
 public:
  template<class Lhv, class Rhv>
  constexpr static inline auto Apply(const Lhv &l,
									 const Rhv &r,
									 std::size_t index) noexcept {
	if constexpr (std::is_arithmetic<Lhv>() &&
		std::is_arithmetic<Rhv>()) {
	  return l * r;
	} else if constexpr (std::is_arithmetic<Lhv>() &&
		!std::is_arithmetic<Rhv>()) {
	  return l * r[index];
	} else if constexpr(!std::is_arithmetic<Lhv>() &&
		std::is_arithmetic<Rhv>()) {
	  return l[index] * r;
	} else {
	  return l[index] * r[index];
	}
  }
};

// -----------------------------------------------------------------------------


class Division {
 public:
  template<class Lhv, class Rhv>
  constexpr static inline auto Apply(const Lhv &l,
									 const Rhv &r,
									 std::size_t index) noexcept {
	if constexpr (std::is_arithmetic<Lhv>() &&
		std::is_arithmetic<Rhv>()) {
	  return l / r;
	} else if constexpr (std::is_arithmetic<Lhv>() &&
		!std::is_arithmetic<Rhv>()) {
	  return l / r[index];
	} else if constexpr(!std::is_arithmetic<Lhv>() &&
		std::is_arithmetic<Rhv>()) {
	  return l[index] / r;
	} else {
	  return l[index] / r[index];
	}
  }
};

// -----------------------------------------------------------------------------

class Minus {
 public:
  template<class Lhv, class Rhv>
  constexpr static inline auto Apply(const Lhv &l,
									 const Rhv &r,
									 std::size_t index) noexcept {
	if constexpr (std::is_arithmetic<Lhv>() &&
		std::is_arithmetic<Rhv>()) {
	  return l - r;
	} else if constexpr (std::is_arithmetic<Lhv>() &&
		!std::is_arithmetic<Rhv>()) {
	  return l - r[index];
	} else if constexpr(!std::is_arithmetic<Lhv>() &&
		std::is_arithmetic<Rhv>()) {
	  return l[index] - r;
	} else {
	  return l[index] - r[index];
	}
  }
};

// -----------------------------------------------------------------------------

}
#endif //NAGATOLIB_SRC_VECTOR_H_
