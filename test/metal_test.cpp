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

constexpr std::size_t array_length = 10;
constexpr std::size_t buffer_size = array_length * sizeof(float);
TEST(MetalTest, MetalTest) {
  NS::AutoreleasePool *pool
    = NS::AutoreleasePool::alloc()->init();
  MTL::Device *device
    = MTL::CreateSystemDefaultDevice();

  MetalFunctionBase metal_base("metal_kernel/add.metal",
                               "add_arrays",
                               array_length,
                               device);

  // buffer を取得
  MetalBuffer<float> buffer_a(device, buffer_size);
  MetalBuffer<float> buffer_b(device, buffer_size);
  MetalBuffer<float> buffer_result(device, buffer_size);

  // buffer にデータを書き込む
  for (std::size_t i = 0; i < array_length; i++)
  {
    buffer_a[i] = static_cast<float>(i);
    buffer_b[i] = static_cast<float>(i);
  }
  // buffer を encoder にセット
  metal_base.SetBuffer(buffer_a, 0, 0);
  metal_base.SetBuffer(buffer_b, 0, 1);
  metal_base.SetBuffer(buffer_result, 0, 2);

  // カーネルを実行
  metal_base.ExecuteKernel();
  std::cout << "Execution finished." << std::endl;

  for (std::size_t i = 0; i < array_length; i++)
  {
    std::cout << buffer_result[i] << ", ";
  }
  std::cout << std::endl;
  metal_base.Reset();

  // buffer にデータを書き込む
  for (std::size_t i = 0; i < array_length; i++)
  {
    buffer_a[i] = static_cast<float>(i) + 1;
    buffer_b[i] = static_cast<float>(i) + 1;
  }

  // buffer を encoder にセット
  metal_base.SetBuffer(buffer_a, 0, 0);
  metal_base.SetBuffer(buffer_b, 0, 1);
  metal_base.SetBuffer(buffer_result, 0, 2);
  metal_base.ExecuteKernel();


  std::cout << "Execution finished." << std::endl;

  for (std::size_t i = 0; i < array_length; i++)
  {
    std::cout << buffer_result[i] << ", ";
  }
  std::cout << std::endl;

  // メモリ解放
  device->autorelease();
  pool->release();
}

#endif // NAGATO_METAL