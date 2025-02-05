//
// Created by toru on 2024/08/29.
//


#include "metal_function_base.hpp"

#include <fstream>
#include <filesystem>
#include <iostream>
#include <utility>

namespace nagato::mtl
{

MetalFunctionBase::MetalFunctionBase(
  MTL::ComputePipelineState *function_pso,
  MTL::Device *p_device,
  MTL::CommandQueue *p_command_queue
  ) :
  device_(NS::RetainPtr(p_device)),
  command_queue_(NS::RetainPtr(p_command_queue)),
  function_pso_(NS::TransferPtr(function_pso))
{
  command_buffer_ = NS::RetainPtr(command_queue_->commandBuffer());
  compute_command_encoder_ = NS::RetainPtr(command_buffer_->computeCommandEncoder());
  compute_command_encoder_->setComputePipelineState(function_pso_->retain());
}


void MetalFunctionBase::ExecuteKernel(MTL::Size grid_size, MTL::Size thread_group_size)
{
  compute_command_encoder_->dispatchThreads(grid_size, thread_group_size);

  compute_command_encoder_->endEncoding();
  command_buffer_->commit();
  command_buffer_->waitUntilCompleted();
}

void MetalFunctionBase::SetBuffer(const MTL::Buffer *buffer, int offset, int index)
{
  compute_command_encoder_->setBuffer(buffer, offset, index);
}

void MetalFunctionBase::Reset()
{
  // 以前のコマンドエンコーダを終了して解放
  if (compute_command_encoder_)
  {
    compute_command_encoder_->endEncoding();
    compute_command_encoder_.reset();
  }

  // 以前のコマンドバッファを解放
  if (command_buffer_)
  {
    command_buffer_.reset();
  }

  // 新しいコマンドバッファとコマンドエンコーダを作成
  command_buffer_ = NS::RetainPtr(command_queue_->commandBuffer());
  compute_command_encoder_ = NS::RetainPtr(command_buffer_->computeCommandEncoder());
  compute_command_encoder_->setComputePipelineState(function_pso_->retain());
}

MetalFunctionBase::~MetalFunctionBase()
{
  // エンコード済みの場合はエンコードを終了
  if (compute_command_encoder_)
  {
    compute_command_encoder_->endEncoding();
  }
}

std::size_t MetalFunctionBase::maxTotalThreadsPerThreadgroup() const
{
  return function_pso_->maxTotalThreadsPerThreadgroup();
}

void MetalFunctionBase::SetThreadgroupMemoryLength(std::size_t length, int index)
{
  this->compute_command_encoder_->setThreadgroupMemoryLength(length, index);
}

} // namespace nagato
