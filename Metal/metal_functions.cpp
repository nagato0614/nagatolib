//
// Created by toru on 2024/08/31.
//

#include "metal_functions.hpp"

namespace nagato::mtl
{
auto &MLASingleton::GetMetalBase()
{
  return metal_base_;
}

MLASingleton::MLASingleton()
{
  metal_base_ = std::make_unique<MetalBase>("metal_kernel/linear_algebra.metal");
}

MetalAdderFunction::MetalAdderFunction(std::size_t length)
  : buffer_length_(length)
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  add_arrays_ = base->CreateFunctionBase("add_arrays");
}

void MetalAdderFunction::operator()(
  const nFloat *inA,
  const nFloat *inB,
  nFloat *result)
{
  // buffer の作成
  auto bufferA = add_arrays_->CreateBuffer<float>(buffer_length_);
  auto bufferB = add_arrays_->CreateBuffer<float>(buffer_length_);
  auto bufferResult = add_arrays_->CreateBuffer<float>(buffer_length_);
  auto bufferLength = add_arrays_->CreateBuffer<uint>(1);
  auto buffer_length = static_cast<uint>(buffer_length_);

  // buffer にデータをコピー
  bufferA.CopyToDevice(inA, buffer_length_);
  bufferB.CopyToDevice(inB, buffer_length_);
  bufferLength.CopyToDevice(&buffer_length, 1);

  // buffer を function にセット
  add_arrays_->SetBuffer(bufferA, 0, 0);
  add_arrays_->SetBuffer(bufferB, 0, 1);
  add_arrays_->SetBuffer(bufferResult, 0, 2);
  add_arrays_->SetBuffer(bufferLength, 0, 3); // buffer_length をカーネルに渡す

  // スレッドグループサイズとグリッドサイズを設定
  const uint threadsPerGroup = DefaultThreadPerGroup;
  uint threadGroups =
    ((buffer_length_ + DataSizePerThread - 1) / DataSizePerThread + threadsPerGroup - 1)
    / threadsPerGroup;

  MTL::Size grid_size = MTL::Size(threadsPerGroup * threadGroups, 1, 1);
  MTL::Size thread_group_size = MTL::Size(threadsPerGroup, 1, 1);

  // 関数の実行
  add_arrays_->ExecuteKernel(grid_size, thread_group_size);

  // 結果をコピー
  bufferResult.CopyToHost(result, buffer_length_);
}

MetalSubFunction::MetalSubFunction(std::size_t length)
  : buffer_length_(length)
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  sub_arrays_ = base->CreateFunctionBase("sub_arrays");
}

void MetalSubFunction::operator()(
  const nFloat *inA,
  const nFloat *inB,
  nFloat *result)
{
  // buffer の作成
  auto bufferA = sub_arrays_->CreateBuffer<float>(buffer_length_);
  auto bufferB = sub_arrays_->CreateBuffer<float>(buffer_length_);
  auto bufferResult = sub_arrays_->CreateBuffer<float>(buffer_length_);

  // buffer にデータをコピー
  bufferA.CopyToDevice(inA, buffer_length_);
  bufferB.CopyToDevice(inB, buffer_length_);

  // buffer を function にセット
  sub_arrays_->SetBuffer(bufferA, 0, 0);
  sub_arrays_->SetBuffer(bufferB, 0, 1);
  sub_arrays_->SetBuffer(bufferResult, 0, 2);

  // 関数の実行
  MTL::Size grid_size = MTL::Size(buffer_length_, 1, 1);
  MTL::Size thread_group_size = MTL::Size(DefaultThreadPerGroup, 1, 1);
  sub_arrays_->ExecuteKernel(grid_size, thread_group_size);

  // 結果をコピー
  bufferResult.CopyToHost(result, buffer_length_);
}

MetalMulFunction::MetalMulFunction(std::size_t length)
  : buffer_length_(length)
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  mul_arrays_ = base->CreateFunctionBase("mul_arrays");
}

void MetalMulFunction::operator()(
  const nFloat *inA,
  const nFloat *inB,
  nFloat *result)
{
  // buffer の作成
  auto bufferA = mul_arrays_->CreateBuffer<float>(buffer_length_);
  auto bufferB = mul_arrays_->CreateBuffer<float>(buffer_length_);
  auto bufferResult = mul_arrays_->CreateBuffer<float>(buffer_length_);

  // buffer にデータをコピー
  bufferA.CopyToDevice(inA, buffer_length_);
  bufferB.CopyToDevice(inB, buffer_length_);

  // buffer を function にセット
  mul_arrays_->SetBuffer(bufferA, 0, 0);
  mul_arrays_->SetBuffer(bufferB, 0, 1);
  mul_arrays_->SetBuffer(bufferResult, 0, 2);

  // 関数の実行
  MTL::Size grid_size = MTL::Size(buffer_length_, 1, 1);
  MTL::Size thread_group_size = MTL::Size(DefaultThreadPerGroup, 1, 1);
  mul_arrays_->ExecuteKernel(grid_size, thread_group_size);

  // 結果をコピー
  bufferResult.CopyToHost(result, buffer_length_);
}

MetalDivFunction::MetalDivFunction(std::size_t length)
  : buffer_length_(length)
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  div_arrays_ = base->CreateFunctionBase("div_arrays");
}

void MetalDivFunction::operator()(
  const nFloat *inA,
  const nFloat *inB,
  nFloat *result)
{
  // buffer の作成
  auto bufferA = div_arrays_->CreateBuffer<float>(buffer_length_);
  auto bufferB = div_arrays_->CreateBuffer<float>(buffer_length_);
  auto bufferResult = div_arrays_->CreateBuffer<float>(buffer_length_);

  // buffer にデータをコピー
  bufferA.CopyToDevice(inA, buffer_length_);
  bufferB.CopyToDevice(inB, buffer_length_);

  // buffer を function にセット
  div_arrays_->SetBuffer(bufferA, 0, 0);
  div_arrays_->SetBuffer(bufferB, 0, 1);
  div_arrays_->SetBuffer(bufferResult, 0, 2);

  // 関数の実行
  MTL::Size grid_size = MTL::Size(buffer_length_, 1, 1);
  MTL::Size thread_group_size = MTL::Size(DefaultThreadPerGroup, 1, 1);
  div_arrays_->ExecuteKernel(grid_size, thread_group_size);

  // 結果をコピー
  bufferResult.CopyToHost(result, buffer_length_);
}

MetalSqrtFunction::MetalSqrtFunction(std::size_t length)
  : buffer_length_(length)
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  sqrt_arrays_ = base->CreateFunctionBase("sqrt_arrays");
}

void MetalSqrtFunction::operator()(const nFloat *inA, nFloat *result)
{
  // buffer の作成
  auto bufferA = sqrt_arrays_->CreateBuffer<float>(buffer_length_);
  auto bufferResult = sqrt_arrays_->CreateBuffer<float>(buffer_length_);

  // buffer にデータをコピー
  bufferA.CopyToDevice(inA, buffer_length_);

  // buffer を function にセット
  sqrt_arrays_->SetBuffer(bufferA, 0, 0);
  sqrt_arrays_->SetBuffer(bufferResult, 0, 1);

  // スレッドグループサイズとグリッドサイズを設定
  const uint threadsPerGroup = DefaultThreadPerGroup;
  uint threadGroups = (buffer_length_ + threadsPerGroup - 1) / threadsPerGroup;

  MTL::Size grid_size = MTL::Size(threadsPerGroup * threadGroups, 1, 1);
  MTL::Size thread_group_size = MTL::Size(threadsPerGroup, 1, 1);

  // 関数の実行
  sqrt_arrays_->ExecuteKernel(grid_size, thread_group_size);

  // 結果をコピー
  bufferResult.CopyToHost(result, buffer_length_);
}

MetalSumFunction::MetalSumFunction(std::size_t length)
  : buffer_length_(length)
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  sum_arrays_ = base->CreateFunctionBase("sum_arrays");
}

void MetalSumFunction::operator()(const nFloat *inA, nFloat *result)
{
  // buffer の作成
  auto bufferA = sum_arrays_->CreateBuffer<float>(buffer_length_);
  auto bufferResult = sum_arrays_->CreateBuffer<float>(1);
  auto buffer_params = sum_arrays_->CreateBuffer<unsigned int>(2);

  // 結果を初期化
  bufferResult[0] = 0.0f;
  buffer_params[0] = static_cast<unsigned int>(this->buffer_length_); // arraySize

  // buffer にデータをコピー
  bufferA.CopyToDevice(inA, buffer_length_);

  // buffer を function にセット
  sum_arrays_->SetBuffer(bufferA, 0, 0);
  sum_arrays_->SetBuffer(bufferResult, 0, 1);
  sum_arrays_->SetBuffer(buffer_params, 0, 2);

  // グループ共有メモリの確保
  //    1次元ディスパッチで、groupCount × threadsPerGroup = totalThreads
  const uint threadsPerGroup = 256;
  const uint groupCount = (buffer_length_ + threadsPerGroup - 1) / threadsPerGroup;
  uint totalThreads = groupCount * threadsPerGroup;
  buffer_params[1] = totalThreads; // totalThreads
  const size_t shared_memory_size = threadsPerGroup * sizeof(float);
  sum_arrays_->SetThreadgroupMemoryLength(shared_memory_size, 0);

  // 実行時のグリッドサイズ & スレッドグループサイズを設定
  MTL::Size grid_size = MTL::Size(totalThreads, 1, 1);
  MTL::Size thread_group_size = MTL::Size(threadsPerGroup, 1, 1);

  // 関数の実行
  sum_arrays_->ExecuteKernel(grid_size, thread_group_size);

  // 結果をコピー
  bufferResult.CopyToHost(result, 1);
}

MetalSoftmaxFunction::MetalSoftmaxFunction(std::size_t arrayLength)
  : array_length_(arrayLength)
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  softmax_ = base->CreateFunctionBase("softmax");
}

void MetalSoftmaxFunction::operator()(const float *inputArray, float *resultArray)
{
  // バッファの作成
  auto bufferInput = softmax_->CreateBuffer<float>(array_length_);
  auto bufferResult = softmax_->CreateBuffer<float>(array_length_);
  auto bufferGlobalSum = softmax_->CreateBuffer<float>(1);
  auto bufferArraySize = softmax_->CreateBuffer<uint>(1);

  // バッファにデータをコピー
  bufferInput.CopyToDevice(inputArray, array_length_);
  bufferArraySize[0] = static_cast<uint>(array_length_);

  // バッファを関数にセット
  softmax_->SetBuffer(bufferInput, 0, 0);
  softmax_->SetBuffer(bufferResult, 0, 1);
  softmax_->SetBuffer(bufferGlobalSum, 0, 2);
  softmax_->SetBuffer(bufferArraySize, 0, 3);

  // スレッドグループサイズとグリッドサイズを設定
  const uint threadsPerGroup = 256;
  const uint groupCount = (array_length_ + threadsPerGroup - 1) / threadsPerGroup;
  uint totalThreads = groupCount * threadsPerGroup;
  bufferArraySize[1] = totalThreads;

  MTL::Size grid_size = MTL::Size(totalThreads, 1, 1);
  MTL::Size thread_group_size = MTL::Size(threadsPerGroup, 1, 1);

  // グループ共有メモリの確保
  const size_t shared_memory_size = threadsPerGroup * sizeof(float);
  softmax_->SetThreadgroupMemoryLength(shared_memory_size, 0);

  // 関数の実行
  softmax_->ExecuteKernel(grid_size, thread_group_size);

  // 結果をコピー
  bufferResult.CopyToHost(resultArray, array_length_);
}

MetalSigmoidFunction::MetalSigmoidFunction(std::size_t arrayLength)
  : array_length_(arrayLength)
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  sigmoid_ = base->CreateFunctionBase("sigmoid_array");
}

void MetalSigmoidFunction::operator()(const float *inputArray, float *resultArray)
{
  // バッファの作成
  auto bufferInput = sigmoid_->CreateBuffer<float>(array_length_);
  auto bufferResult = sigmoid_->CreateBuffer<float>(array_length_);

  // バッファにデータをコピー
  bufferInput.CopyToDevice(inputArray, array_length_);

  // バッファを関数にセット
  sigmoid_->SetBuffer(bufferInput, 0, 0);
  sigmoid_->SetBuffer(bufferResult, 0, 1);

  // 実行
  MTL::Size grid_size = MTL::Size(array_length_, 1, 1);
  MTL::Size thread_group_size = MTL::Size(DefaultThreadPerGroup, 1, 1);
  sigmoid_->ExecuteKernel(grid_size, thread_group_size);

  // 結果をコピー
  bufferResult.CopyToHost(resultArray, array_length_);
}

MetalReluFunction::MetalReluFunction(std::size_t arrayLength)
  : array_length_(arrayLength)
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  relu_ = base->CreateFunctionBase("relu_array");
}

void MetalReluFunction::operator()(
  const float *inputArray, 
  float *resultArray
  )
{
  // バッファの作成
  auto bufferInput = relu_->CreateBuffer<float>(array_length_);
  auto bufferResult = relu_->CreateBuffer<float>(array_length_);
  auto bufferArraySize = relu_->CreateBuffer<uint>(1);

  // バッファにデータをコピー
  bufferInput.CopyToDevice(inputArray, array_length_);
  bufferArraySize[0] = static_cast<uint>(array_length_);

  // バッファを関数にセット
  relu_->SetBuffer(bufferInput, 0, 0);
  relu_->SetBuffer(bufferResult, 0, 1);
  relu_->SetBuffer(bufferArraySize, 0, 2);

  // 実行
  MTL::Size grid_size = MTL::Size(array_length_, 1, 1);
  MTL::Size thread_group_size = MTL::Size(DefaultThreadPerGroup, 1, 1);
  relu_->ExecuteKernel(grid_size, thread_group_size);

  // 結果をコピー
  bufferResult.CopyToHost(resultArray, array_length_);
}

MetalMatMulFunction::MetalMatMulFunction(std::size_t n, std::size_t m, std::size_t l)
{
  n_ = n;
  m_ = m;
  l_ = l;

  auto &base = MLASingleton::GetInstance().GetMetalBase();
  matmul_ = base->CreateFunctionBase("matmul");
}

} // namespace nagato::mtl
