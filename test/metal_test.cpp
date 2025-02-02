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

void sqrt_array(const float *a, float *result, std::size_t length)
{
  for (std::size_t i = 0; i < length; i++)
  {
    result[i] = std::sqrt(a[i]);
  }
}

void matmul_array(
  const float *a,
  const float *b,
  float *result,
  std::size_t n,
  std::size_t m,
  std::size_t l
)
{
  for (std::size_t i = 0; i < n; i++)
  {
    for (std::size_t j = 0; j < l; j++)
    {
      result[i * l + j] = 0.0f;
    }
  }

  for (std::size_t i = 0; i < n; i++)
  {
    for (std::size_t j = 0; j < l; j++)
    {
      for (std::size_t k = 0; k < m; k++)
      {
        result[i * l + j] += a[i * m + k] * b[k * l + j];
      }
    }
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

  nagato::mtl::MetalAdderFunction adder(array_length);
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

  nagato::mtl::MetalSubFunction sub(array_length);
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

  nagato::mtl::MetalMulFunction mul(array_length);
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

  nagato::mtl::MetalDivFunction div(array_length);
  div(a.get(), b.get(), gpu_result.get());

  for (std::size_t i = 0; i < array_length; i++)
  {
    ASSERT_EQ(cpu_result[i], gpu_result[i]);
  }
}

TEST(MetalTest, sum_arrays)
{
  // GPU を使わない場合の計算
  std::unique_ptr<float[]> a(new float[array_length]);
  std::unique_ptr<float[]> cpu_result(new float[1]);
  std::unique_ptr<float[]> gpu_result(new float[1]);

  // メルセンヌ・ツイスター法による乱数生成器
  std::mt19937 mt(0);

  for (std::size_t i = 0; i < array_length; i++)
  {
    a[i] = static_cast<float>(mt() / mt.max());
  }

  sum_arrays(a.get(), cpu_result.get(), array_length);

  nagato::mtl::MetalSumFunction sum(array_length);
  sum(a.get(), gpu_result.get());

  ASSERT_EQ(cpu_result[0], gpu_result[0]);
}

TEST(MetalTest, sqrt_array)
{
  // GPU を使わない場合の計算
  std::unique_ptr<float[]> a(new float[array_length]);
  std::unique_ptr<float[]> cpu_result(new float[array_length]);
  std::unique_ptr<float[]> gpu_result(new float[array_length]);

  // メルセンヌ・ツイスター法による乱数生成器
  std::mt19937 mt(0);

  for (std::size_t i = 0; i < array_length; i++)
  {
    a[i] = static_cast<float>(mt());
  }

  sqrt_array(a.get(), cpu_result.get(), array_length);

  nagato::mtl::MetalSqrtFunction sqrt(array_length);
  sqrt(a.get(), gpu_result.get());

  for (std::size_t i = 0; i < array_length; i++)
  {
    const auto diff = std::abs(cpu_result[i] - gpu_result[i]);
    ASSERT_LT(diff, 1e-2);
  }
}

TEST(MetalTest, matmul_array)
{

  constexpr std::size_t n = 1000;
  constexpr std::size_t m = 1000;
  constexpr std::size_t l = 1000;

  std::vector<float> a;
  std::vector<float> b;

  // メルセンヌツイスターで乱数を生成
  std::mt19937 mt(0);
  for (std::size_t i = 0; i < n * m; i++)
  {
    a.push_back(static_cast<float>(mt()));
  }

  for (std::size_t i = 0; i < m * l; i++)
  {
    b.push_back(static_cast<float>(mt()));
  }

  std::vector<float> cpu_result(n * l, 0.0f);
  std::vector<float> gpu_result(n * l, 0.0f);

  // 初期化
  for (std::size_t i = 0; i < n * l; i++)
  {
    cpu_result[i] = 0.0f;
    gpu_result[i] = 0.0f;
  }

  // CPUで行列積を計算
  const auto start_cpu = std::chrono::system_clock::now();
  matmul_array(a.data(), b.data(), cpu_result.data(), n, m, l);
  const auto end_cpu = std::chrono::system_clock::now();
  const auto elapsed_cpu =
    std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();

  // GPUで行列積を計算
  const auto start_gpu = std::chrono::system_clock::now();
  nagato::mtl::MetalMatMulFunction metal_matmul_function(n, m, l);
  metal_matmul_function(a.data(), b.data(), gpu_result.data());
  const auto end_gpu = std::chrono::system_clock::now();
  const auto elapsed_gpu =
    std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();

  // 誤差をチェック
  for (std::size_t i = 0; i < n * l; i++)
  {
    if (std::abs(cpu_result[i] - gpu_result[i]) > 1e-2)
    {
      std::cerr << "Error: matmul result[" << i << "] = " << cpu_result[i] << " vs " << gpu_result[i]
                << std::endl;
    }

    if (i > 10)
    {
      break;
    }
  }
}


#endif // NAGATO_METAL