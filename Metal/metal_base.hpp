//
// Created by toru on 2024/08/31.
//

#ifndef NAGATOLIB_METAL_METAL_BASE_HPP_
#define NAGATOLIB_METAL_METAL_BASE_HPP_

#include <Metal.hpp>
#include "metal_buffer.hpp"

class MetalBase
{
 public:
  explicit MetalBase();

  ~MetalBase();

  [[nodiscard]] MTL::Device *GetDevice() const
  {
    return device_->retain();
  }

  [[nodiscard]] MTL::CommandQueue *GetCommandQueue() const
  {
    return command_queue_->retain();
  }

  /**
   * Buffer を作成してmoveして返す
   * @tparam T
   * @param buffer_size
   * @return
   */
  template<typename T>
  [[nodiscard]] MetalBuffer<T> CreateBuffer(std::size_t buffer_length = 1)
  {
    MetalBuffer<T> buffer(device_.get(), buffer_length);
    return buffer;
  }

 private:

  NS::AutoreleasePool *pool_;
  NS::SharedPtr<MTL::Device> device_;
  NS::SharedPtr<MTL::CommandQueue> command_queue_;
};

#endif //NAGATOLIB_METAL_METAL_BASE_HPP_
