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

#include "metal_function_base.hpp"
#include "metal_buffer.hpp"
#include "metal_base.hpp"

constexpr std::size_t array_length = 11;

void add_arrays(const float *a, const float *b, float *result, std::size_t length)
{
  for (std::size_t i = 0; i < length; i++)
  {
    result[i] = a[i] + b[i];
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
  nagato::MetalBase metal_base("metal_kernel/linear_algebra.metal");

  auto metal_function_base
    = metal_base.CreateFunctionBase("add_arrays",
                                    array_length);

  // buffer を取得
  auto buffer_a = metal_function_base->CreateBuffer<float>();
  auto buffer_b = metal_function_base->CreateBuffer<float>();
  auto buffer_result = metal_function_base->CreateBuffer<float>();
  buffer_a.ShowBufferSize();

  // buffer にデータを書き込む
  for (std::size_t i = 0; i < array_length; i++)
  {
    buffer_a[i] = static_cast<float>(i);
    buffer_b[i] = static_cast<float>(i);
  }
  // buffer を encoder にセット
  metal_function_base->SetBuffer(buffer_a, 0, 0);
  metal_function_base->SetBuffer(buffer_b, 0, 1);
  metal_function_base->SetBuffer(buffer_result, 0, 2);

  // カーネルを実行
  metal_function_base->ExecuteKernel();

  // GPU を使わない場合の計算時間を計測
  std::unique_ptr<float[]> a(new float[array_length]);
  std::unique_ptr<float[]> b(new float[array_length]);
  std::unique_ptr<float[]> result(new float[array_length]);

  for (std::size_t i = 0; i < array_length; i++)
  {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i);
  }

  add_arrays(a.get(), b.get(), result.get(), array_length);

  for (std::size_t i = 0; i < array_length; i++)
  {
    ASSERT_EQ(result[i], buffer_result[i]);
  }
}

TEST(MetalTest, sum_arrays)
{
  nagato::MetalBase metal_base("metal_kernel/linear_algebra.metal");

  auto metal_function_base
    = metal_base.CreateFunctionBase("sum_arrays",
                                    array_length);


  // buffer を取得
  auto buffer_a = metal_function_base->CreateBuffer<float>();
  auto buffer_result = metal_function_base->CreateBuffer<float>(1);
  buffer_a.ShowBufferSize();

  // buffer にデータを書き込む
  for (std::size_t i = 0; i < array_length; i++)
  {
    buffer_a[i] = static_cast<float>(i);
  }
  // buffer を encoder にセット
  metal_function_base->SetBuffer(buffer_a, 0, 0);
  metal_function_base->SetBuffer(buffer_result, 0, 1);

  // カーネルを実行
  metal_function_base->ExecuteKernel();

  // GPU を使わない場合の計算時間を計測
  std::unique_ptr<float[]> a(new float[array_length]);
  std::unique_ptr<float> result(new float[array_length]);

  for (std::size_t i = 0; i < array_length; i++)
  {
    a[i] = static_cast<float>(i);
  }

  sum_arrays(a.get(), result.get(), array_length);

  ASSERT_EQ(*result, buffer_result[0]);
}

#endif // NAGATO_METAL