//
// Created by toru on 2024/08/31.
//

#include "metal_base.hpp"
MetalBase::MetalBase()
{
  pool_ = NS::AutoreleasePool::alloc()->init();
  device_ = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  command_queue_ = NS::TransferPtr(device_->newCommandQueue());
}

MetalBase::~MetalBase()
{
  pool_->release();
}
