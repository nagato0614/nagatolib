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

constexpr std::size_t array_length = 10;
constexpr std::size_t buffer_size = array_length * sizeof(float);
int main()
{
  nagato::MetalBase metal_base;

  nagato::MetalFunctionBase metal_function_base("metal_kernel/add.metal",
                                        "add_arrays",
                                        array_length,
                                        metal_base.GetDevice(),
                                        metal_base.GetCommandQueue());

  // buffer を取得
  auto buffer_a = metal_base.CreateBuffer<float>(array_length);
  auto buffer_b = metal_base.CreateBuffer<float>(array_length);
  auto buffer_result = metal_base.CreateBuffer<float>(array_length);

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

  // カーネルを実行
  metal_function_base.ExecuteKernel();
  std::cout << "Execution finished." << std::endl;

  for (std::size_t i = 0; i < array_length; i++)
  {
    std::cout << buffer_result[i] << ", ";
  }
  std::cout << std::endl;
  metal_function_base.Reset();

  // buffer にデータを書き込む
  for (std::size_t i = 0; i < array_length; i++)
  {
    buffer_a[i] = static_cast<float>(i) + 1;
    buffer_b[i] = static_cast<float>(i) + 1;
  }

  // buffer を encoder にセット
  metal_function_base.SetBuffer(buffer_a, 0, 0);
  metal_function_base.SetBuffer(buffer_b, 0, 1);
  metal_function_base.SetBuffer(buffer_result, 0, 2);
  metal_function_base.ExecuteKernel();

  std::cout << "Execution finished." << std::endl;

  for (std::size_t i = 0; i < array_length; i++)
  {
    std::cout << buffer_result[i] << ", ";
  }
  std::cout << std::endl;

  return 0;
}