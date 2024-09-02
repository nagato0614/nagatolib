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

MetalFunctionBase::MetalFunctionBase(std::size_t buffer_length,
                                     MTL::ComputePipelineState *function_pso,
                                     MTL::Device *p_device,
                                     MTL::CommandQueue *p_command_queue) :
  buffer_length_(buffer_length),
  device_(NS::RetainPtr(p_device)),
  command_queue_(NS::RetainPtr(p_command_queue)),
  function_pso_(NS::TransferPtr(function_pso))
{
  command_buffer_ = NS::RetainPtr(command_queue_->commandBuffer());
  compute_command_encoder_ = NS::RetainPtr(command_buffer_->computeCommandEncoder());
  compute_command_encoder_->setComputePipelineState(function_pso_->retain());
}

void MetalFunctionBase::ExecuteKernel()
{
  MTL::Size grid_size = MTL::Size(buffer_length_, 1, 1);

  NS::UInteger thread_group_size_ = function_pso_->maxTotalThreadsPerThreadgroup();
  std::cout << "thread_group_size: " << thread_group_size_ << std::endl;
  if (thread_group_size_ > buffer_length_)
  {
    thread_group_size_ = buffer_length_;
  }

  MTL::Size thread_group_size = MTL::Size(thread_group_size_, 1, 1);

  this->ExecuteKernel(grid_size, thread_group_size);
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


} // namespace nagato
