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

constexpr std::size_t array_length = 1000 * 1000;

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

void sqrt_arrays(const float *a, float *result, std::size_t length)
{
  for (std::size_t i = 0; i < length; i++)
  {
    result[i] = std::sqrt(a[i]);
  }
}

void softmax_arrays(const float *a, float *result, std::size_t length)
{
    if (length == 0) return;

    float max_val = *std::max_element(a, a + length);

    float sum = 0.0f;
    for (std::size_t i = 0; i < length; i++)  
    {
        result[i] = expf(a[i] - max_val) + 1e-30f;  // underflow 防止
        sum += result[i];
    }

    float inv_sum = 1.0f / sum;
    for (std::size_t i = 0; i < length; i++)
    {
        result[i] *= inv_sum;
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

  std::cout << "Elapsed time with CPU : " << elapsed_cpu << " us" << std::endl;
  std::cout << "Elapsed time with GPU : " << elapsed << " us" << std::endl;

  delete[] a;
  delete[] b;
  delete[] result;
  delete[] gpu_result;
}

void sum_example()
{
  // 入力配列をCPU側で用意
  auto a = std::make_unique<float[]>(array_length);
  {
    std::random_device rnd;
    for (std::size_t i = 0; i < array_length; i++)
    {
      a[i] = 1.0f / array_length;
    }
  }

  // CPUで総和を計算 & 時間計測
  float cpu_sum = 0.0f;
  {
    const auto start_cpu = std::chrono::system_clock::now();
    for (std::size_t i = 0; i < array_length; i++)
    {
      cpu_sum += a[i];
    }
    const auto end_cpu = std::chrono::system_clock::now();
    const auto elapsed_cpu =
      std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();

    std::cout << "[CPU] sum=" << cpu_sum
              << " , time=" << elapsed_cpu << " us" << std::endl;
  }

  nagato::mtl::MetalSumFunction metal_sum_function(array_length);

  float gpu_sum = 0.0f;
  // GPU でカーネルを実行 & 時間計測
  const auto start_gpu = std::chrono::system_clock::now();
  metal_sum_function(a.get(), &gpu_sum);
  const auto end_gpu = std::chrono::system_clock::now();
  const auto elapsed_gpu =
    std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();

  // 結果を表示
  std::cout << "[GPU] sum=" << gpu_sum
            << " , time=" << elapsed_gpu << " us" << std::endl;

  // 誤差をチェック
  float diff = std::abs(cpu_sum - gpu_sum);
  if (diff > 1.0f) // 値の大きさ次第で多少の誤差は出やすいため、判定をある程度緩めに
  {
    std::cerr << "Error: CPU sum=" << cpu_sum
              << " vs GPU sum=" << gpu_sum
              << " (diff=" << diff << ")" << std::endl;
  }
  else
  {
    std::cout << "OK: diff=" << diff << std::endl;
  }
}

void sqrt_example()
{
  nagato::mtl::MLASingleton::GetInstance();

  // GPU を使わない場合の計算時間を計測
  auto *a = new float[array_length];
  auto *result = new float[array_length];
  auto *gpu_result = new float[array_length];

  std::mt19937 mt(0);
  for (std::size_t i = 0; i < array_length; i++)
  {
    a[i] = static_cast<float>(mt());
    result[i] = 0.f;
    gpu_result[i] = 0.f;
  }

  const auto start_cpu = std::chrono::system_clock::now();
  sqrt_arrays(a, result, array_length);
  const auto end_cpu = std::chrono::system_clock::now();
  const auto elapsed_cpu =
    std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();

  // 計算時間を計測
  const auto start = std::chrono::system_clock::now();

  nagato::mtl::MetalSqrtFunction metal_sqrt_function(array_length);
  metal_sqrt_function(a, gpu_result);

  // カーネルを実行
  const auto end = std::chrono::system_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  for (std::size_t i = 0; i < array_length; i++)
  {
    if (std::abs(result[i] - gpu_result[i]) > 1.f)
    {
      std::cerr << "Error: sqrt result[" << i << "] = " << result[i] << " vs " << gpu_result[i]
                << std::endl;
    }
  }

  std::cout << "Elapsed time with CPU : " << elapsed_cpu << " us" << std::endl;
  std::cout << "Elapsed time with GPU : " << elapsed << " us" << std::endl;

  delete[] a;
  delete[] result;
  delete[] gpu_result;
}

void softmax_example()
{
  // 入力配列をCPU側で用意
  auto a = std::make_unique<float[]>(array_length);
  auto cpu_result = std::make_unique<float[]>(array_length);
  auto gpu_result = std::make_unique<float[]>(array_length);
  {
    std::random_device rnd;
    for (std::size_t i = 0; i < array_length; i++)
    {
      a[i] = static_cast<float>(rnd());
    }
  }

  // CPUでソフトマックスを計算 & 時間計測
  const auto start_cpu = std::chrono::system_clock::now();
  softmax_arrays(a.get(), cpu_result.get(), array_length);
  const auto end_cpu = std::chrono::system_clock::now();
  const auto elapsed_cpu =
    std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();

  // GPUでソフトマックスを計算 & 時間計測
  const auto start_gpu = std::chrono::system_clock::now();
  nagato::mtl::MetalSoftmaxFunction metal_softmax_function(array_length);
  metal_softmax_function(a.get(), gpu_result.get());
  const auto end_gpu = std::chrono::system_clock::now();
  const auto elapsed_gpu =
    std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();

  // 誤差をチェック
  for (std::size_t i = 0; i < array_length; i++)
  {
    if (std::abs(cpu_result[i] - gpu_result[i]) > 1e-2)
    {
      std::cerr << "Error: softmax result[" << i << "] = " << cpu_result[i] << " vs " << gpu_result[i]
                << std::endl;
    }
  }

  std::cout << "Elapsed time with CPU : " << elapsed_cpu << " us" << std::endl;
  std::cout << "Elapsed time with GPU : " << elapsed_gpu << " us" << std::endl;
}

int main()
{
  // std::cout << "--- add_example ---" << std::endl;
  // add_example();

  // std::cout << "--- sum_example ---" << std::endl;
  // sum_example();

  // std::cout << "--- sqrt_example ---" << std::endl;
  // sqrt_example();

  std::cout << "--- softmax_example ---" << std::endl;
  softmax_example();

  return 0;
}