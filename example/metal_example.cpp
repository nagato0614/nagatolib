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

void sqrt_arrays(const float *a, float *result, std::size_t length)
{
  for (std::size_t i = 0; i < length; i++)
  {
    result[i] = std::sqrt(a[i]);
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

void sum_example_2()
{
  // 1) 入力配列をCPU側で用意
  auto a = std::make_unique<float[]>(array_length);
  {
    std::random_device rnd;
    for (std::size_t i = 0; i < array_length; i++)
    {
      // 0～1範囲の乱数
      a[i] = 1.f;
    }
  }

  // 2) CPUで総和を計算 & 時間計測
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

  // 3) Metal の準備
  //    linear_algebra.metal 内に定義したカーネル "sum_arrays_full" をロード
  nagato::mtl::MetalBase metal_base("../metal_kernel/linear_algebra.metal");
  auto metal_function_base
    = metal_base.CreateFunctionBase("sum_arrays_full", array_length);

  // 4) バッファの作成
  //    入力用バッファ (配列a)
  auto buffer_a = metal_function_base->CreateBuffer<float>(array_length);
  buffer_a.CopyToDevice(a.get(), array_length);

  //    原子加算 (atomic_uint) 用のバッファ (要素1)
  //    カーネル内で最終的な合計値をビット変換して加算していく
  auto buffer_globalSum = metal_function_base->CreateBuffer<unsigned int>(1);
  buffer_globalSum[0] = 0; // 初期値 0

  //    カーネルに渡すパラメータ [arraySize, totalThreads]
  auto buffer_params = metal_function_base->CreateBuffer<unsigned int>(2);
  buffer_params[0] = static_cast<unsigned int>(array_length); // arraySize

  // 5) スレッドグループとスレッド総数の計算
  //    1次元ディスパッチで、groupCount × threadsPerGroup = totalThreads
  const uint threadsPerGroup = 256;
  const uint groupCount = (array_length + threadsPerGroup - 1) / threadsPerGroup;
  uint totalThreads = groupCount * threadsPerGroup;
  buffer_params[1] = totalThreads; // totalThreads

  // 6) 作成したバッファをカーネルに紐付け
  //    ( index = 0,1,2 は sum_arrays_fullカーネルの定義に合わせる )
  metal_function_base->SetBuffer(buffer_a,         0, 0);
  metal_function_base->SetBuffer(buffer_globalSum, 0, 1);
  metal_function_base->SetBuffer(buffer_params,    0, 2);

  // グループ共有メモリの確保
  const size_t shared_memory_size = threadsPerGroup * sizeof(float);
  metal_function_base->SetThreadgroupMemoryLength(shared_memory_size, 0);

  // 7) 実行時のグリッドサイズ & スレッドグループサイズを設定
  MTL::Size grid_size         = MTL::Size(totalThreads, 1, 1);
  MTL::Size thread_group_size = MTL::Size(threadsPerGroup, 1, 1);

  // 8) GPU でカーネルを実行 & 時間計測
  const auto start_gpu = std::chrono::system_clock::now();
  metal_function_base->ExecuteKernel(grid_size, thread_group_size);
  const auto end_gpu = std::chrono::system_clock::now();
  const auto elapsed_gpu =
    std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();

  // 9) 結果 (atomic_uint) を float に再変換
  auto bits = buffer_globalSum[0];
  float gpu_sum = *reinterpret_cast<float *>(&bits);

  // 10) 結果を表示
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
    a[i] = 3.f;
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
    if (std::abs(result[i] - gpu_result[i]) > 1e-5)
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

int main()
{
  std::cout << "--- add_example ---" << std::endl;
  add_example();

  std::cout << "--- sum_example_2 ---" << std::endl;
  sum_example_2();

  std::cout << "--- sqrt_example ---" << std::endl;
  sqrt_example();
  return 0;
}