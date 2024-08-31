//
// Created by toru on 2024/08/29.
//
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal.hpp>

#include <iostream>

#include "metal_base.hpp"
#include "metal_function_base.hpp"

void add_arrays(const float *a, const float *b, float *result, std::size_t length)
{
  for (std::size_t i = 0; i < length; i++)
  {
    result[i] = a[i] + b[i];
  }
}

constexpr std::size_t array_length = 100000000;
int main()
{
  nagato::MetalBase metal_base;

  auto metal_function_base
    = metal_base.CreateFunctionBase("metal_kernel/linear_algebra.metal",
                                    "add_arrays",
                                    array_length);

  // buffer を取得
  auto buffer_a = metal_function_base.CreateBuffer<float>();
  auto buffer_b = metal_function_base.CreateBuffer<float>();
  auto buffer_result = metal_function_base.CreateBuffer<float>();
  buffer_a.ShowBufferSize();

  // buffer にデータを書き込む
  for (std::size_t i = 0; i < array_length; i++)
  {
    buffer_a[i] = static_cast<float>(i);
    buffer_b[i] = static_cast<float>(i);
  }
  // buffer を encoder にセット
  metal_function_base.SetBuffer(buffer_a, 0, 0);
  metal_function_base.SetBuffer(buffer_b, 0, 1);
  metal_function_base.SetBuffer(buffer_result, 0, 2);

  // 計算時間を計測
  const auto start = std::chrono::system_clock::now();
  // カーネルを実行
  metal_function_base.ExecuteKernel();
  const auto end = std::chrono::system_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();


  // GPU を使わない場合の計算時間を計測
  auto *a = new float[array_length];
  auto *b = new float[array_length];
  auto *result = new float[array_length];

  for (std::size_t i = 0; i < array_length; i++)
  {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i);
  }

  const auto start_cpu = std::chrono::system_clock::now();
  add_arrays(a, b, result, array_length);
  const auto end_cpu = std::chrono::system_clock::now();
  const auto elapsed_cpu = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();

  for (std::size_t i = 0; i < array_length; i++)
  {
    if (result[i] != buffer_result[i])
    {
      std::cerr << "Error: result[" << i << "] = " << result[i] << " vs " << buffer_result[i] << std::endl;
    }
  }

  std::cout << "Elapsed time with GPU : " << elapsed << " us" << std::endl;
  std::cout << "Elapsed time with CPU : " << elapsed_cpu << " us" << std::endl;

  delete[] a;
  delete[] b;
  delete[] result;

  return 0;
}