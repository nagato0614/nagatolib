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
// -----------------------------------------------------------------------------
// forward declaration
template<typename Primitive, std::size_t size>
class Vector;

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
	using __self = Vector<Primitive, size>;
	using _array_type = std::array<Primitive, size>;
	using _size = std::size_t;
	using _reference = __self &;
	using _const_reference = const __self &;
	using _rvalue_reference = __self &&;

	constexpr
	explicit Vector(Primitive p = 0.0) noexcept {
		for (auto &i : array_)
			i = p;
	}

	constexpr Vector(const std::initializer_list<Primitive> &init) noexcept {
		assert(init.size() <= size);
		for (_size i = 0; i < init.size(); i++)
			array_[i] = static_cast<Primitive>(*(init.begin() + i));
	}

	constexpr Vector(const std::vector<Primitive> &v) noexcept {
		assert(v.size() <= size);
		for (_size i = 0; i < v.size(); i++)
			array_[i] = v[i];
	}

	/**
	 * copy constructor
	 * @param v
	 */
	constexpr Vector(_const_reference v) noexcept
			: array_(v.array_) {}

	/**
	 * move constructor
	 * @param v
	 */
	constexpr Vector(_rvalue_reference v) noexcept
			: array_(v.array_) {}

	~Vector() = default;

	constexpr __self &operator=(_const_reference v) noexcept {
		array_ = v.array_;
		return *this;
	}

	constexpr __self &operator+=(_const_reference v) noexcept {
		for (int i = 0; i < size; i++)
			array_[i] += v[i];
		return *this;
	}

	template<typename T>
	constexpr __self &operator+=(const T &value) noexcept {
		STATIC_ASSERT_IS_ARITHMETRIC(T);
		for (int i = 0; i < size; i++)
			array_[i] += value;
		return *this;
	}

	constexpr __self &operator-=(_const_reference v) noexcept {
		for (int i = 0; i < size; i++)
			array_[i] -= v[i];
		return *this;
	}

	template<typename T>
	constexpr __self &operator-=(const T &value) noexcept {
		STATIC_ASSERT_IS_ARITHMETRIC(T);
		for (int i = 0; i < size; i++)
			array_[i] -= value;
		return *this;
	}

	constexpr __self &operator*=(_const_reference v) noexcept {
		for (int i = 0; i < size; i++)
			array_[i] *= v[i];
		return *this;
	}

	template<typename T>
	constexpr __self &operator*=(const T &value) noexcept {
		STATIC_ASSERT_IS_ARITHMETRIC(T);
		for (int i = 0; i < size; i++)
			array_[i] *= value;
		return *this;
	}

	constexpr __self &operator/=(_const_reference v) noexcept {
		assert(!v.HasZero());
		for (int i = 0; i < size; i++)
			array_[i] /= v[i];
		return *this;
	}

	template<typename T>
	constexpr __self &operator/=(const T &value) noexcept {
		STATIC_ASSERT_IS_ARITHMETRIC(T);
		for (int i = 0; i < size; i++)
			array_[i] /= value;
		return *this;
	}

	constexpr Primitive &operator[](_size index)
	& noexcept {
		assert(0 <= index && index < size);
		return array_[index];
	}

	constexpr const Primitive &operator[](_size index)
	const & noexcept {
		assert(0 <= index && index < size);
		return array_[index];
	}

	constexpr Primitive operator[](_size index)
	const && noexcept {
		assert(0 <= index && index < size);
		return array_[index];
	}

	constexpr void Sqrt() noexcept {
		for (_size i = 0; i < size; i++)
			array_[i] = sqrt(array_[i]);
	}

	constexpr _size GetArraySize()
	const noexcept {
		return size;
	}

	constexpr bool HasNan()
	const noexcept {
		for (const auto &i : array_)
			if (is_nan(i))
				return true;
		return false;
	}

	constexpr bool HasZero()
	const noexcept {
		for (const auto &i : array_)
			if (i == static_cast<Primitive>(array_[i]))
				return true;
		return false;
	}

	constexpr Primitive Max()
	const noexcept {
		Primitive m = 0;
		for (const auto &i : array_)
			m = max(m, i);
		return m;
	}

	constexpr Primitive Min()
	const noexcept {
		Primitive m = 0;
		for (const auto &i : array_)
			m = min(m, i);
		return m;
	}

	template<typename Type>
	constexpr void Fill(Type t) noexcept {
		STATIC_ASSERT_IS_ARITHMETRIC(Type);
		for (auto &i : array_)
			i = static_cast<Primitive>(t);
	}

	template<typename L, typename R>
	constexpr void Clamp(L l, R r) noexcept {
		for (auto &i : array_)
			i = Clamp(i, l, r);
	}

	template<typename F>
	constexpr __self itor(
			const __self &value,
			F &&f
	) const noexcept {
		for (auto &i : array_)
			i = f(i);
	}

	constexpr Primitive Sum()
	const noexcept {
		Primitive sum = 0.0;
		for (const auto &i : array_)
			sum += i;
		return sum;
	}

	constexpr auto begin() const noexcept {
		return array_.begin();
	}

	constexpr auto end() const noexcept {
		return array_.end();
	}
 private:
	_array_type array_ = {0};
};

// -----------------------------------------------------------------------------

template<typename Primitive, std::size_t size>
constexpr Vector<Primitive, size> operator+(
		const Vector<Primitive, size> &lhv,
		const Vector<Primitive, size> &rhv
) noexcept {
	return Vector(lhv) += rhv;
}

template<typename Primitive, std::size_t size, typename T>
constexpr Vector<Primitive, size> operator+(
		const Vector<Primitive, size> &lhv,
		const T &rhv
) noexcept {
	STATIC_ASSERT_IS_ARITHMETRIC(T);
	return Vector(lhv) += rhv;
}

template<typename Primitive, std::size_t size, typename T>
constexpr Vector<Primitive, size> operator+(
		const T &lhv,
		const Vector<Primitive, size> &rhv
) noexcept {
	STATIC_ASSERT_IS_ARITHMETRIC(T);
	return Vector(rhv) += lhv;
}

// -----------------------------------------------------------------------------


template<typename Primitive, std::size_t size>
constexpr Vector<Primitive, size> operator-(
		const Vector<Primitive, size> &lhv,
		const Vector<Primitive, size> &rhv
) noexcept {
	return Vector(lhv) -= rhv;
}

template<typename Primitive, std::size_t size, typename T>
constexpr Vector<Primitive, size> operator-(
		const Vector<Primitive, size> &lhv,
		const T &rhv
) noexcept {
	STATIC_ASSERT_IS_ARITHMETRIC(T);
	return Vector(lhv) -= rhv;
}

template<typename Primitive, std::size_t size, typename T>
constexpr Vector<Primitive, size> operator-(
		const T &lhv,
		const Vector<Primitive, size> &rhv
) noexcept {
	STATIC_ASSERT_IS_ARITHMETRIC(T);
	return Vector(rhv) -= lhv;
}
// -----------------------------------------------------------------------------


template<typename Primitive, std::size_t size>
constexpr Vector<Primitive, size> operator*(
		const Vector<Primitive, size> &lhv,
		const Vector<Primitive, size> &rhv
) noexcept {
	return Vector(lhv) *= rhv;
}

template<typename Primitive, std::size_t size, typename T>
constexpr Vector<Primitive, size> operator*(
		const Vector<Primitive, size> &lhv,
		const T &rhv
) noexcept {
	STATIC_ASSERT_IS_ARITHMETRIC(T);
	return Vector(lhv) *= rhv;
}

template<typename Primitive, std::size_t size, typename T>
constexpr Vector<Primitive, size> operator*(
		const T &lhv,
		const Vector<Primitive, size> &rhv
) noexcept {
	STATIC_ASSERT_IS_ARITHMETRIC(T);
	return Vector(rhv) *= lhv;
}

// -----------------------------------------------------------------------------

template<typename Primitive, std::size_t size>
constexpr Vector<Primitive, size> operator/(
		const Vector<Primitive, size> &lhv,
		const Vector<Primitive, size> &rhv
) noexcept {
	return Vector(lhv) /= rhv;
}

template<typename Primitive, std::size_t size, typename T>
constexpr Vector<Primitive, size> operator/(
		const Vector<Primitive, size> &lhv,
		const T &rhv
) noexcept {
	STATIC_ASSERT_IS_ARITHMETRIC(T);
	return Vector(lhv) /= rhv;
}

template<typename Primitive, std::size_t size, typename T>
constexpr Vector<Primitive, size> operator/(
		const T &lhv,
		const Vector<Primitive, size> &rhv
) noexcept {
	STATIC_ASSERT_IS_ARITHMETRIC(T);
	Vector<Primitive, size> result;
	for (std::size_t i = 0; i < size; i++)
		result[i] = lhv / rhv[i];
	return result;
}

// -----------------------------------------------------------------------------

template<typename Primitive, std::size_t size>
constexpr Primitive Dot(
		const Vector<Primitive, size> &lhv,
		const Vector<Primitive, size> &rhv
) noexcept {
	return (lhv * rhv).Sum();
}

template<typename Primitive, std::size_t size>
constexpr Vector<Primitive, size> Sqrt(
		const Vector<Primitive, size> &value
) noexcept {
	Vector<Primitive, size> v(value);
	for (std::size_t i = 0; i < size; i++)
		v[i] = sqrt(value[i]);
	return v;
}

// -----------------------------------------------------------------------------

template<class Char, class Traits, class Primitive, std::size_t size>
std::basic_ostream<Char, Traits> &operator<<(
		std::basic_ostream<Char, Traits> &os,
		const Vector<Primitive, size> &dt) {
	os << "[";

	for (int i = 0; i < size - 1; i++)
		os << dt[i] << ", ";
	os << dt[size - 1];

	os << "]";
	return os;
}

}
#endif //NAGATOLIB_SRC_VECTOR_H_
