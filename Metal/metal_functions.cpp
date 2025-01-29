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
  metal_base_ = std::make_unique<MetalBase>("../metal_kernel/linear_algebra.metal");
}

MetalAdderFunction::MetalAdderFunction(std::size_t length)
  : buffer_length_(length)
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  add_arrays_ = base->CreateFunctionBase("add_arrays", buffer_length_);
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
  uint threadGroups = ((buffer_length_ + DataSizePerThread - 1) / DataSizePerThread + threadsPerGroup - 1) / threadsPerGroup;

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
  sub_arrays_ = base->CreateFunctionBase("sub_arrays", buffer_length_);
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
  sub_arrays_->ExecuteKernel();

  // 結果をコピー
  bufferResult.CopyToHost(result, buffer_length_);
}

MetalMulFunction::MetalMulFunction(std::size_t length)
  : buffer_length_(length)
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  mul_arrays_ = base->CreateFunctionBase("mul_arrays", buffer_length_);
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
  mul_arrays_->ExecuteKernel();

  // 結果をコピー
  bufferResult.CopyToHost(result, buffer_length_);
}

MetalDivFunction::MetalDivFunction(std::size_t length)
  : buffer_length_(length)
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  div_arrays_ = base->CreateFunctionBase("div_arrays", buffer_length_);
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
  div_arrays_->ExecuteKernel();

  // 結果をコピー
  bufferResult.CopyToHost(result, buffer_length_);
}

MetalSqrtFunction::MetalSqrtFunction(std::size_t length)
  : buffer_length_(length)
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  sqrt_arrays_ = base->CreateFunctionBase("sqrt_arrays", buffer_length_);
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

} // namespace nagato::mtl
