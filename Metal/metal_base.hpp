//
// Created by toru on 2024/08/31.
//

#ifndef NAGATOLIB_METAL_METAL_BASE_HPP_
#define NAGATOLIB_METAL_METAL_BASE_HPP_

#include "Metal.hpp"
#include "metal_buffer.hpp"
#include "metal_function_base.hpp"

namespace nagato
{
class MetalBase
{
 public:
  explicit MetalBase();

  ~MetalBase();

  /**
   * デバイスを取得. 所有権は呼び出し側にある
   * @return
   */
  [[nodiscard]] MTL::Device *GetDevice() const
  {
    return device_->retain();
  }

  /**
   * コマンドキューを取得. 所有権は呼び出し側にある
   * @return
   */
  [[nodiscard]] MTL::CommandQueue *GetCommandQueue() const
  {
    return command_queue_->retain();
  }



  [[nodiscard]] MetalFunctionBase CreateFunctionBase(std::string kernel_file_name,
                                                     std::string kernel_function_name,
                                                     std::size_t buffer_length);

 private:

  NS::AutoreleasePool *pool_;
  NS::SharedPtr<MTL::Device> device_;
  NS::SharedPtr<MTL::CommandQueue> command_queue_;
};

} // namespace nagato
#endif //NAGATOLIB_METAL_METAL_BASE_HPP_
