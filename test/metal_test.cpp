//
// Created by toru on 2024/08/30.
//

#include <gtest/gtest.h>

#ifdef NAGATO_METAL

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal.hpp>
#include <iostream>
#include <random>

#include "metal_function_base.hpp"
#include "metal_buffer.hpp"
#include "metal_base.hpp"
#include "metal_functions.hpp"
#include "metal_base.hpp"

constexpr std::size_t array_length = 1000;

void add_arrays(const float *a, const float *b, float *result, std::size_t length)
{
  for (std::size_t i = 0; i < length; i++)
  {
    result[i] = a[i] + b[i];
  }
}

void sub_arrays(const float *a, const float *b, float *result, std::size_t length)
{
  for (std::size_t i = 0; i < length; i++)
  {
    result[i] = a[i] - b[i];
  }
}

void mul_arrays(const float *a, const float *b, float *result, std::size_t length)
{
  for (std::size_t i = 0; i < length; i++)
  {
    result[i] = a[i] * b[i];
  }
}

void div_arrays(const float *a, const float *b, float *result, std::size_t length)
{
  for (std::size_t i = 0; i < length; i++)
  {
    result[i] = a[i] / b[i];
  }
}

void sum_arrays(const float *a, float *result, std::size_t length)
{
  for (std::size_t i = 0; i < length; i++)
  {
    *result += a[i];
  }
}

TEST(MetalTest, add_arrays)
{
  // GPU を使わない場合の計算
  std::unique_ptr<float[]> a(new float[array_length]);
  std::unique_ptr<float[]> b(new float[array_length]);
  std::unique_ptr<float[]> cpu_result(new float[array_length]);

  std::unique_ptr<float[]> gpu_result(new float[array_length]);

  // メルセンヌ・ツイスター法による乱数生成器
  std::mt19937 mt(0);

  for (std::size_t i = 0; i < array_length; i++)
  {
    a[i] = static_cast<float>(mt() / mt.max());
    b[i] = static_cast<float>(mt() / mt.max());
  }

  add_arrays(a.get(), b.get(), cpu_result.get(), array_length);

  nagato::mla::MetalAdderFunction adder(array_length);
  adder(a.get(), b.get(), gpu_result.get());

  for (std::size_t i = 0; i < array_length; i++)
  {
    ASSERT_EQ(cpu_result[i], gpu_result[i]);
  }
}

TEST(MetalTest, sub_arrays)
{
  // GPU を使わない場合の計算
  std::unique_ptr<float[]> a(new float[array_length]);
  std::unique_ptr<float[]> b(new float[array_length]);
  std::unique_ptr<float[]> cpu_result(new float[array_length]);
  std::unique_ptr<float[]> gpu_result(new float[array_length]);

  // メルセンヌ・ツイスター法による乱数生成器
  std::mt19937 mt(0);

  for (std::size_t i = 0; i < array_length; i++)
  {
    a[i] = static_cast<float>(mt() / mt.max());
    b[i] = static_cast<float>(mt() / mt.max());
  }

  sub_arrays(a.get(), b.get(), cpu_result.get(), array_length);

  nagato::mla::MetalSubFunction sub(array_length);
  sub(a.get(), b.get(), gpu_result.get());

  for (std::size_t i = 0; i < array_length; i++)
  {
    ASSERT_EQ(cpu_result[i], gpu_result[i]);
  }
}

TEST(MetalTest, mul_arrays)
{
  // GPU を使わない場合の計算
  std::unique_ptr<float[]> a(new float[array_length]);
  std::unique_ptr<float[]> b(new float[array_length]);
  std::unique_ptr<float[]> cpu_result(new float[array_length]);

  std::unique_ptr<float[]> gpu_result(new float[array_length]);

  // メルセンヌ・ツイスター法による乱数生成器
  std::mt19937 mt(0);

  for (std::size_t i = 0; i < array_length; i++)
  {
    a[i] = static_cast<float>(mt() / mt.max());
    b[i] = static_cast<float>(mt() / mt.max());
  }

  mul_arrays(a.get(), b.get(), cpu_result.get(), array_length);

  nagato::mla::MetalMulFunction mul(array_length);
  mul(a.get(), b.get(), gpu_result.get());

  for (std::size_t i = 0; i < array_length; i++)
  {
    ASSERT_EQ(cpu_result[i], gpu_result[i]);
  }
}

TEST(MetalTest, div_arrays)
{
  // GPU を使わない場合の計算
  std::unique_ptr<float[]> a(new float[array_length]);
  std::unique_ptr<float[]> b(new float[array_length]);
  std::unique_ptr<float[]> cpu_result(new float[array_length]);

  std::unique_ptr<float[]> gpu_result(new float[array_length]);

  // メルセンヌ・ツイスター法による乱数生成器
  std::mt19937 mt(0);

  for (std::size_t i = 0; i < array_length; i++)
  {
    a[i] = static_cast<float>(mt() / mt.max());
    b[i] = static_cast<float>(mt() / mt.max()) + 1e-6;
  }

  div_arrays(a.get(), b.get(), cpu_result.get(), array_length);

  nagato::mla::MetalDivFunction div(array_length);
  div(a.get(), b.get(), gpu_result.get());

  for (std::size_t i = 0; i < array_length; i++)
  {
    ASSERT_EQ(cpu_result[i], gpu_result[i]);
  }
}



#endif // NAGATO_METAL