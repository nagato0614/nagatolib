#include <gtest/gtest.h>

#include "vector.hpp"
#include "random.hpp"

using namespace nagato;

template<typename Primitive, std::size_t size>
std::vector<Primitive> MakeVector()
{
  std::random_device random_device;
  Random rnd(random_device());
  std::vector<Primitive> v;
  v.reserve(size);
  for (int i = 0; i < size; i++)
  {
    v.push_back(rnd.uniform_real_distribution<Primitive>(-100.0, 100));
  }
  return v;
}

template<typename Primitive, std::size_t size>
std::vector<Primitive> MakeVectorHasNonZero()
{
  std::random_device random_device;
  Random rnd(random_device());
  std::vector<Primitive> v;
  v.reserve(size);
  for (int i = 0; i < size; i++)
  {
    auto random_number = rnd.uniform_real_distribution<Primitive>(-100.0, 100);
    while (random_number == 0)
    {
      random_number = rnd.uniform_real_distribution<Primitive>(-100.0, 100);
    }
    random_number = rnd.uniform_real_distribution<Primitive>(-100.0, 100);
    v.push_back(random_number);
  }
  return v;
}

template<typename Primitive, std::size_t size>
auto plus(
  const std::vector<Primitive> &a,
  const std::vector<Primitive> &b
)
{
  std::vector<Primitive> v;
  v.reserve(size);
  for (int i = 0; i < size; i++)
    v.push_back(a[i] + b[i]);
  return v;
}

template<typename Primitive, std::size_t size>
std::vector<Primitive> minus(
  const std::vector<Primitive> &a,
  const std::vector<Primitive> &b
)
{
  std::vector<Primitive> v;
  v.reserve(size);
  for (int i = 0; i < size; i++)
  {
    v.push_back(a[i] - b[i]);
  }
  return v;
}

template<typename Primitive, std::size_t size>
std::vector<Primitive> multi(
  const std::vector<Primitive> &a,
  const std::vector<Primitive> &b
)
{
  std::vector<Primitive> v;
  for (int i = 0; i < size; i++)
  {
    v.emplace_back(a[i] * b[i]);
  }
  return v;
}

template<typename Primitive, std::size_t size>
std::vector<Primitive> division(
  const std::vector<Primitive> &a,
  const std::vector<Primitive> &b
)
{
  std::vector<Primitive> v;
  for (int i = 0; i < size; i++)
  {
    v.emplace_back(a[i] / b[i]);
  }
  return v;
}

template<typename Primitive, std::size_t size>
void assert_eq(
  const std::vector<Primitive> &a,
  const Vector<Primitive, size> &b
)
{
  for (int i = 0; i < size; i++)
    ASSERT_EQ(a[i], b[i]);
}

TEST(VECTOR, PLUS)
{
  //1 test param
  constexpr int array_size = 100;
  using Primitive = double;
  using Vector3d = Vector<Primitive, array_size>;
  constexpr int MAX_TEST_CASE = 10000;

  std::random_device random_device;
  Random rnd(random_device());

  std::vector<std::vector<Primitive>> test_case_1;
  std::vector<std::vector<Primitive>> test_case_2;

  // make vectors
  for (int i = 0; i < MAX_TEST_CASE; i++)
  {
    test_case_1.push_back(MakeVector<Primitive, array_size>());
    test_case_2.push_back(MakeVector<Primitive, array_size>());
  }

  // test Arithmetic operations


  for (int i = 0; i < MAX_TEST_CASE; i++)
  {
    auto ans = plus<Primitive, array_size>(test_case_1[i], test_case_2[i]);

    Vector3d a{test_case_1[i]};
    Vector3d b{test_case_2[i]};
    Vector3d a_b = a + b;

    assert_eq(ans, a_b);
  }
}

TEST(VECTOR, MINUS)
{
  //1 test param
  constexpr int array_size = 100;
  using Primitive = double;
  using Vector3d = Vector<Primitive, array_size>;
  constexpr int MAX_TEST_CASE = 10000;

  std::random_device random_device;
  Random rnd(random_device());

  std::vector<std::vector<Primitive>> test_case_1;
  std::vector<std::vector<Primitive>> test_case_2;

  // make vectors
  for (int i = 0; i < MAX_TEST_CASE; i++)
  {
    test_case_1.push_back(MakeVector<Primitive, array_size>());
    test_case_2.push_back(MakeVector<Primitive, array_size>());
  }

  // test Arithmetic operations


  for (int i = 0; i < MAX_TEST_CASE; i++)
  {
    auto ans = minus<Primitive, array_size>(test_case_1[i], test_case_2[i]);

    Vector3d a{test_case_1[i]};
    Vector3d b{test_case_2[i]};
    Vector3d a_b = a - b;

    assert_eq(ans, a_b);
  }
}

TEST(VECTOR, MULTI)
{
  //1 test param
  constexpr int array_size = 100;
  using Primitive = double;
  using Vector3d = Vector<Primitive, array_size>;
  constexpr int MAX_TEST_CASE = 10000;

  std::random_device random_device;
  Random rnd(random_device());

  std::vector<std::vector<Primitive>> test_case_1;
  std::vector<std::vector<Primitive>> test_case_2;

  // make vectors
  for (int i = 0; i < MAX_TEST_CASE; i++)
  {
    test_case_1.push_back(MakeVector<Primitive, array_size>());
    test_case_2.push_back(MakeVector<Primitive, array_size>());
  }

  // test Arithmetic operations


  for (int i = 0; i < MAX_TEST_CASE; i++)
  {
    auto ans = multi<Primitive, array_size>(test_case_1[i], test_case_2[i]);

    Vector3d a{test_case_1[i]};
    Vector3d b{test_case_2[i]};
    Vector3d a_b = a * b;

    assert_eq(ans, a_b);
  }
}

TEST(VECTOR, Division)
{
  //1 test param
  constexpr int array_size = 100;
  using Primitive = double;
  using Vector3d = Vector<Primitive, array_size>;
  constexpr int MAX_TEST_CASE = 10000;

  std::random_device random_device;
  Random rnd(random_device());

  std::vector<std::vector<Primitive>> test_case_1;
  std::vector<std::vector<Primitive>> test_case_2;

  // make vectors
  for (int i = 0; i < MAX_TEST_CASE; i++)
  {
    test_case_1.push_back(MakeVector<Primitive, array_size>());
    test_case_2.push_back(MakeVectorHasNonZero<Primitive, array_size>());
  }

  // test Arithmetic operations


  for (int i = 0; i < MAX_TEST_CASE; i++)
  {
    auto ans = division<Primitive, array_size>(test_case_1[i], test_case_2[i]);

    Vector3d a{test_case_1[i]};
    Vector3d b{test_case_2[i]};
    Vector3d a_b = a / b;

    assert_eq(ans, a_b);
  }
}