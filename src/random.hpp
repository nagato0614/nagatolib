//
// Created by nagato0614 on 2019-06-24.
//

#pragma once
#ifndef NAGATOLIB_SRC_RANDOM_H_
#define NAGATOLIB_SRC_RANDOM_H_

#include <random>
#include "assert.hpp"

namespace nagato
{

/**
 * stdのrandom engineをラップするクラス
 * @tparam Engine 使用したいエンジン
 */
template<typename Engine = std::mt19937>
class Random
{
 public:
  explicit Random(std::size_t seed = 0) noexcept: engine_(seed)
  {}
  ~Random() = default;

  template<class Return, class From, class To>
  Return uniform_int_distribution(From from, To to) noexcept
  {
	static_assert(std::is_arithmetic<Return>(),
				  "Return is not arithmetic");
	std::uniform_int_distribution<Return> distribution(from, to);
	return distribution(engine_);
  }

  template<typename Float>
  auto uniform_real_distribution(Float from, Float to) noexcept
  {
	static_assert(std::is_arithmetic<Float>(),
				  "Float is not arithmetic");
	std::uniform_real_distribution<Float> distribution(from, to);
	return distribution(engine_);
  }

 private:
  Engine engine_;
};

template<typename UINT = std::uint64_t>
class LinearCongruential
{
  STATIC_ASSERT_IS_ARITHMETRIC(UINT);
  static_assert(std::is_unsigned<UINT>(),
				"UINT is signed type!");

  using ld = long double;
 public:
  constexpr LinearCongruential(UINT seed = 1) noexcept
	  : seed_(seed), x_(seed)
  {
  }

  constexpr UINT seed() const noexcept
  {
	return seed_;
  }

  constexpr UINT next() noexcept
  {
	return x_ = (coefficient_ * x_) % mod_;
  }

  template<typename Int, typename From, typename To>
  constexpr Int uniform_int(From from, To to) noexcept
  {
	STATIC_ASSERT_IS_INTEGER(Int);
	ld diff = static_cast<From>(to) - from;
	return from + diff * next() / mod_;
  }

  template<typename Real, typename From, typename To>
  constexpr Real uniform_real(From from, To to) noexcept
  {
	STATIC_ASSERT_IS_FLOATING_POINT(Real);
	From diff = static_cast<From>(to) - from;
	return from + diff * (next() / mod_);
  }

 private:
  UINT x_;
  const UINT coefficient_ = 48271;
  const UINT mod_ = std::numeric_limits<UINT>::max();
  const UINT seed_;
};

}

#endif //NAGATOLIB_SRC_RANDOM_H_,
