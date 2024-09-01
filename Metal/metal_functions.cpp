//
// Created by toru on 2024/08/31.
//

#include "metal_functions.hpp"

namespace nagato::mla
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
  nFloat *result
)
{
  // buffer の作成
  auto bufferA = add_arrays_->CreateBuffer<float>(buffer_length_);
  auto bufferB = add_arrays_->CreateBuffer<float>(buffer_length_);
  auto bufferResult = add_arrays_->CreateBuffer<float>(buffer_length_);

  // buffer にデータをコピー
  bufferA.CopyToDevice(inA, buffer_length_);
  bufferB.CopyToDevice(inB, buffer_length_);

  // buffer を function にセット
  add_arrays_->SetBuffer(bufferA, 0, 0);
  add_arrays_->SetBuffer(bufferB, 0, 1);
  add_arrays_->SetBuffer(bufferResult, 0, 2);

  // 関数の実行
  add_arrays_->ExecuteKernel();

  // 結果をコピー
  bufferResult.CopyToHost(result, buffer_length_);
}

MetalSubFunction::MetalSubFunction(std::size_t length)
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  sub_arrays_ = base->CreateFunctionBase("sub_arrays", length);
}

void MetalSubFunction::operator()(
  const nFloat *inA,
  const nFloat *inB,
  nFloat *result
)
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
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  mul_arrays_ = base->CreateFunctionBase("mul_arrays", length);
}

void MetalMulFunction::operator()(
  const nFloat *inA,
  const nFloat *inB,
  nFloat *result
)
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
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  div_arrays_ = base->CreateFunctionBase("div_arrays", length);
}

void MetalDivFunction::operator()(
  const nFloat *inA,
  const nFloat *inB,
  nFloat *result
)
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


} // namespace nagato::mla