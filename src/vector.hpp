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
#include "type_traits.hpp"

namespace nagato
{
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
 * @tparam Size vectorに保存するデータ数
 */
template<typename Primitive, std::size_t Size>
class Vector
{
  static_assert(std::is_arithmetic<Primitive>(),
                "Primitive Type is not Arithmetric!");

 public:
  using self = Vector<Primitive, Size>;
  using array_type = std::array<Primitive, Size>;
  using size = std::size_t;
  using reference = self &;
  using const_reference = const self &;
  using rvalue_reference = self &&;

  constexpr
  explicit Vector(Primitive p = 0.0) noexcept
  {
    for (auto &i : array_)
      i = p;
  }

  constexpr Vector(const std::initializer_list<Primitive> &init) noexcept
  {
    assert(init.size() <= Size);
    for (size i = 0; i < init.size(); i++)
      array_[i] = static_cast<Primitive>(*(init.begin() + i));
  }

  template<typename T>
  constexpr explicit Vector(const std::vector<T> &v) noexcept
  {
    assert(v.size() <= Size);
    for (size i = 0; i < v.size(); i++)
      array_[i] = static_cast<Primitive>(v[i]);
  }

  /**
   * copy constructor
   * @param v
   */
  constexpr Vector(const_reference v) noexcept
    : array_(v.array_)
  {}

  /**
   * move constructor
   * @param v
   */
  constexpr Vector(rvalue_reference v) noexcept
    : array_(std::move(v.array_))
  {}

  template<typename... Args,
    typename std::enable_if<is_all_arithmetic<Args...>::value>::type *
    = nullptr>
  constexpr explicit Vector(Args &&... args) noexcept
  {
    SetData(std::forward<Args>(args)...);
  }

  template<typename... Args>
  constexpr void SetData(Args &&... args) noexcept
  {
    constexpr std::size_t param_size = sizeof...(Args);
    static_assert(param_size <= Size);
    array_ = {static_cast<Primitive>(args)...};
  }

  ~Vector() = default;

  constexpr self &operator=(const_reference v) noexcept
  {
    array_ = v.array_;
    return *this;
  }

  constexpr self &operator+=(const_reference v) noexcept
  {
    for (int i = 0; i < Size; i++)
      array_[i] += v[i];
    return *this;
  }

  template<typename T>
  constexpr self &operator+=(const T &value) noexcept
  {
    STATIC_ASSERT_IS_ARITHMETRIC(T);
    for (int i = 0; i < Size; i++)
      array_[i] += value;
    return *this;
  }

  constexpr self &operator-=(const_reference v) noexcept
  {
    for (int i = 0; i < Size; i++)
      array_[i] -= v[i];
    return *this;
  }

  template<typename T>
  constexpr self &operator-=(const T &value) noexcept
  {
    STATIC_ASSERT_IS_ARITHMETRIC(T);
    for (int i = 0; i < Size; i++)
      array_[i] -= value;
    return *this;
  }

  constexpr self &operator*=(const_reference v) noexcept
  {
    for (int i = 0; i < Size; i++)
      array_[i] *= v[i];
    return *this;
  }

  template<typename T>
  constexpr self &operator*=(const T &value) noexcept
  {
    STATIC_ASSERT_IS_ARITHMETRIC(T);
    for (int i = 0; i < Size; i++)
      array_[i] *= value;
    return *this;
  }

  constexpr self &operator/=(const_reference v) noexcept
  {
    assert(!v.HasZero());
    for (int i = 0; i < Size; i++)
      array_[i] /= v[i];
    return *this;
  }

  template<typename T>
  constexpr self &operator/=(const T &value) noexcept
  {
    STATIC_ASSERT_IS_ARITHMETRIC(T);
    for (int i = 0; i < Size; i++)
      array_[i] /= value;
    return *this;
  }

  constexpr self &operator-() noexcept
  {
    for (auto &i : array_)
      i *= -1;
    return *this;
  }

  constexpr Primitive &operator[](size index)
  & noexcept
  {
    assert(0 <= index && index < Size);
    return array_[index];
  }

  constexpr const Primitive &operator[](size index)
  const & noexcept
  {
    assert(0 <= index && index < Size);
    return array_[index];
  }

  constexpr Primitive operator[](size index)
  const && noexcept
  {
    assert(0 <= index && index < Size);
    return array_[index];
  }

  constexpr void Sqrt() noexcept
  {
    for (size i = 0; i < Size; i++)
      array_[i] = sqrt(array_[i]);
  }

  constexpr size GetArraySize()
  const noexcept
  {
    return Size;
  }

  constexpr bool HasNan()
  const noexcept
  {
    for (const auto &i : array_)
      if (is_nan(i))
      {
        return true;
      }
    return false;
  }

  constexpr bool HasZero()
  const noexcept
  {
    for (const auto &i : array_)
      if (i == static_cast<Primitive>(array_[i]))
      {
        return true;
      }
    return false;
  }

  constexpr Primitive Max()
  const noexcept
  {
    Primitive m = 0;
    for (const auto &i : array_)
      m = max(m, i);
    return m;
  }

  constexpr Primitive Min()
  const noexcept
  {
    Primitive m = 0;
    for (const auto &i : array_)
      m = min(m, i);
    return m;
  }

  template<typename Type>
  constexpr void Fill(Type t) noexcept
  {
    STATIC_ASSERT_IS_ARITHMETRIC(Type);
    for (auto &i : array_)
      i = static_cast<Primitive>(t);
  }

  template<typename L, typename R>
  constexpr self Clamp(L l, R r) noexcept
  {
    STATIC_ASSERT_IS_ARITHMETRIC(L);
    STATIC_ASSERT_IS_ARITHMETRIC(R);
    self result;
    for (size i = 0; i < Size; i++)
    {
      result[i] = clamp(array_[i], l, r);
    }
    return result;
  }

  /**
   * 関数を適用した結果を返す
   * 参照を返すので注意
   * @tparam F
   * @param f
   * @return
   */
  template<typename F>
  constexpr self itor(F &&f) const & noexcept
  {
    self result;
    for (size i = 0; i < Size; i++)
      result[i] = f(array_[i]);
    return result;
  }

  constexpr Primitive Sum()
  const noexcept
  {
    Primitive sum = 0.0;
    for (const auto &i : array_)
      sum += i;
    return sum;
  }

  constexpr Primitive Length() const
  {
    Primitive sum = 0.0;
    for (const auto &i : array_)
      sum += i * i;
    return sum;
  }

  Primitive LengthSqrt() const
  {
    return std::sqrt(Length());
  }
  constexpr auto begin() const noexcept
  {
    return array_.begin();
  }

  constexpr auto end() const noexcept
  {
    return array_.end();
  }

  constexpr auto ToStdVector() const noexcept
  {
    std::vector<Primitive> v;
    for (const auto &i : array_)
      v.push_back(i);
    return v;
  }
 private:
  array_type array_;
};

// -----------------------------------------------------------------------------

template<typename Primitive, std::size_t size>
constexpr Vector<Primitive, size> operator+(
  const Vector<Primitive, size> &lhv,
  const Vector<Primitive, size> &rhv
) noexcept
{
  return Vector(lhv) += rhv;
}

template<typename Primitive, std::size_t size, typename T>
constexpr Vector<Primitive, size> operator+(
  const Vector<Primitive, size> &lhv,
  const T &rhv
) noexcept
{
  STATIC_ASSERT_IS_ARITHMETRIC(T);
  return Vector(lhv) += rhv;
}

template<typename Primitive, std::size_t size, typename T>
constexpr Vector<Primitive, size> operator+(
  const T &lhv,
  const Vector<Primitive, size> &rhv
) noexcept
{
  STATIC_ASSERT_IS_ARITHMETRIC(T);
  return Vector(rhv) += lhv;
}

// -----------------------------------------------------------------------------


template<typename Primitive, std::size_t size>
constexpr Vector<Primitive, size> operator-(
  const Vector<Primitive, size> &lhv,
  const Vector<Primitive, size> &rhv
) noexcept
{
  return Vector(lhv) -= rhv;
}

template<typename Primitive, std::size_t size, typename T>
constexpr Vector<Primitive, size> operator-(
  const Vector<Primitive, size> &lhv,
  const T &rhv
) noexcept
{
  STATIC_ASSERT_IS_ARITHMETRIC(T);
  return Vector(lhv) -= rhv;
}

// -----------------------------------------------------------------------------


template<typename Primitive, std::size_t size>
constexpr Vector<Primitive, size> operator*(
  const Vector<Primitive, size> &lhv,
  const Vector<Primitive, size> &rhv
) noexcept
{
  return Vector(lhv) *= rhv;
}

template<typename Primitive, std::size_t size, typename T>
constexpr Vector<Primitive, size> operator*(
  const Vector<Primitive, size> &lhv,
  const T &rhv
) noexcept
{
  STATIC_ASSERT_IS_ARITHMETRIC(T);
  return Vector(lhv) *= rhv;
}

template<typename Primitive, std::size_t size, typename T>
constexpr Vector<Primitive, size> operator*(
  const T &lhv,
  const Vector<Primitive, size> &rhv
) noexcept
{
  STATIC_ASSERT_IS_ARITHMETRIC(T);
  return Vector(rhv) *= lhv;
}

// -----------------------------------------------------------------------------

template<typename Primitive, std::size_t size>
constexpr Vector<Primitive, size> operator/(
  const Vector<Primitive, size> &lhv,
  const Vector<Primitive, size> &rhv
) noexcept
{
  return Vector(lhv) /= rhv;
}

template<typename Primitive, std::size_t size, typename T>
constexpr Vector<Primitive, size> operator/(
  const Vector<Primitive, size> &lhv,
  const T &rhv
) noexcept
{
  STATIC_ASSERT_IS_ARITHMETRIC(T);
  return Vector(lhv) /= rhv;
}

template<typename Primitive, std::size_t size, typename T>
constexpr Vector<Primitive, size> operator/(
  const T &lhv,
  const Vector<Primitive, size> &rhv
) noexcept
{
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
) noexcept
{
  return (lhv * rhv).Sum();
}

template<typename Primitive>
constexpr Vector<Primitive, 3> Cross(
  const Vector<Primitive, 3> &lhv,
  const Vector<Primitive, 3> &rhv
) noexcept
{
  return Vector<Primitive, 3>({lhv[1] * rhv[2] - lhv[2] * rhv[1],
                               lhv[2] * rhv[0] - lhv[0] * rhv[2],
                               lhv[0] * rhv[1] - lhv[1] * rhv[0]});
}

template<typename Primitive, std::size_t size>
constexpr Vector<Primitive, size> Sqrt(
  const Vector<Primitive, size> &value
) noexcept
{
  Vector<Primitive, size> v(value);
  for (std::size_t i = 0; i < size; i++)
    v[i] = sqrt(value[i]);
  return v;
}

template<typename Primitive, std::size_t size>
constexpr Vector<Primitive, size> Normalize(
  const Vector<Primitive, size> &value
) noexcept
{
  return value / value.LengthSqrt();
}

// -----------------------------------------------------------------------------

template<class Char, class Traits, class Primitive, std::size_t size>
std::basic_ostream<Char, Traits> &operator<<(
  std::basic_ostream<Char, Traits> &os,
  const Vector<Primitive, size> &dt)
{
  os << "[";

  for (int i = 0; i < size - 1; i++)
    os << dt[i] << ", ";
  os << dt[size - 1];

  os << "]";
  return os;
}

// -----------------------------------------------------------------------------

template<typename Primitive, std::size_t size>
constexpr
Vector<Primitive, size> LineSpace(auto min,
                                  auto max) noexcept
{
  STATIC_ASSERT_IS_ARITHMETRIC(Primitive);

  Vector<Primitive, size> result;
  Primitive step = (max - min) / (size - 1);
  for (std::size_t i = 0; i < size; i++)
    result[i] = min + step * i;
  return result;
}

}
#endif //NAGATOLIB_SRC_VECTOR_H_
