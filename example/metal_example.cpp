//
// Created by toru on 2024/08/29.
//
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal.hpp>

#include <iostream>
#include <random>

#include "metal_base.hpp"
#include "metal_common.hpp"
#include "metal_functions.hpp"

constexpr std::size_t array_length = 1980 * 1080 * 3;

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

void add_example()
{
  nagato::mtl::MLASingleton::GetInstance();

  // GPU を使わない場合の計算時間を計測
  auto *a = new float[array_length];
  auto *b = new float[array_length];
  auto *result = new float[array_length];
  auto *gpu_result = new float[array_length];

  std::mt19937 mt(0);
  for (std::size_t i = 0; i < array_length; i++)
  {
    a[i] = static_cast<float>(mt());
    b[i] = static_cast<float>(mt());
  }

  const auto start_cpu = std::chrono::system_clock::now();
  add_arrays(a, b, result, array_length);
  const auto end_cpu = std::chrono::system_clock::now();
  const auto elapsed_cpu =
    std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();


  // 計算時間を計測
  const auto start = std::chrono::system_clock::now();

  nagato::mtl::MetalAdderFunction metal_adder_function(array_length);
  metal_adder_function(a, b, gpu_result);

  // カーネルを実行
  const auto end = std::chrono::system_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  for (std::size_t i = 0; i < array_length; i++)
  {
    if (std::abs(result[i] - gpu_result[i]) > 1e-5)
    {
      std::cerr << "Error: result[" << i << "] = " << result[i] << " vs " << gpu_result[i]
                << std::endl;
    }
  }

  std::cout << "Elapsed time with GPU : " << elapsed << " us" << std::endl;
  std::cout << "Elapsed time with CPU : " << elapsed_cpu << " us" << std::endl;

  delete[] a;
  delete[] b;
  delete[] result;
  delete[] gpu_result;
}

void sum_example()
{
  // CPUで計算
  auto a = std::make_unique<float[]>(array_length);
  for (std::size_t i = 0; i < array_length; i++)
  {
    // ランダムに値を設定
    std::random_device rnd;
    a[i] = static_cast<float>(rnd()) / static_cast<float>(rnd.max());
  }

  float cpu_sum = 0;
  const auto start_cpu = std::chrono::system_clock::now();
  sum_arrays(a.get(), &cpu_sum, array_length);
  const auto end_cpu = std::chrono::system_clock::now();
  const auto elapsed_cpu =
    std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();

  nagato::mtl::MetalBase metal_base("../metal_kernel/linear_algebra.metal");


  // 計算時間を計測
  const auto start = std::chrono::system_clock::now();
  auto metal_function_base
    = metal_base.CreateFunctionBase("sum_arrays",
                                    array_length);

  // buffer を取得
  auto buffer_a = metal_function_base->CreateBuffer<float>();
  auto buffer_array_size = metal_function_base->CreateBuffer<unsigned int>();
  buffer_a.ShowBufferSize();

  // buffer にデータを書き込む
  buffer_a.CopyToDevice(a.get(), array_length);
  buffer_array_size[0] = array_length;

  // buffer を encoder にセット
  metal_function_base->SetBuffer(buffer_a, 0, 0);
  metal_function_base->SetBuffer(buffer_array_size, 0, 2);

  // グリッドサイズとスレッドグループサイズを設定
  const auto
    max_total_threads_per_threadgroup = metal_function_base->maxTotalThreadsPerThreadgroup();
  std::cout << "max_total_threads_per_threadgroup: " << max_total_threads_per_threadgroup
            << std::endl;

  const std::size_t thread_num = (array_length - 1) / data_size_per_thread + 1;
  std::cout << "thread_num: " << thread_num << std::endl;
  const std::size_t thread_group_num = (thread_num - 1) / max_total_threads_per_threadgroup + 1;
  std::cout << "thread_group_num: " << thread_group_num << std::endl;

  const MTL::Size grid_size = MTL::Size(thread_num, 1, 1);
  const MTL::Size thread_group_size = MTL::Size(thread_group_num, 1, 1);

  // 作成するgroupサイズに合わせてresultのバッファーを作成
  auto buffer_result =
    metal_function_base->CreateBuffer<float>(thread_num);
  metal_function_base->SetBuffer(buffer_result, 0, 1);

  // カーネルを実行
  metal_function_base->ExecuteKernel(grid_size, thread_group_size);

  float gpu_sum = 0;
  for (std::size_t i = 0; i < thread_num; i++)
  {
    gpu_sum += buffer_result[i];
  }
  const auto end = std::chrono::system_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  // 結果を表示
  std::cout << "GPU Result: " << gpu_sum << std::endl;
  std::cout << "CPU Result: " << cpu_sum << std::endl;
  std::cout << "Elapsed time with GPU : " << elapsed << " us" << std::endl;
  std::cout << "Elapsed time with CPU : " << elapsed_cpu << " us" << std::endl;

  // 結果を比較
  if (std::abs(gpu_sum - cpu_sum) > 1)
  {
    std::cerr << "Error: diff = " << std::abs(gpu_sum - cpu_sum) << std::endl;
  }
}

int main()
{
  std::cout << "--- add_example ---" << std::endl;
  add_example();

  return 0;
}