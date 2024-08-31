//
// Created by toru on 2024/08/30.
//

#ifndef NAGATOLIB_METAL_METAL_BUFFER_HPP_
#define NAGATOLIB_METAL_METAL_BUFFER_HPP_

#include "Metal.hpp"

namespace nagato{


template<typename T>
class MetalBuffer
{
 public:
  explicit MetalBuffer(MTL::Device *device,
                       std::size_t buffer_length = 1,
                       MTL::ResourceOptions options = MTL::StorageModeShared)
  noexcept: buffer_length_(buffer_length)
  {
    buffer_ = NS::TransferPtr(
      device->newBuffer(
        buffer_length_ * sizeof(T),
        options
      )
    );
  }

  /**
   * コピーコンストラクタ
   * @param other
   */
  MetalBuffer(const MetalBuffer &other) noexcept: buffer_length_(other.buffer_length_)
  {
    buffer_ = other.buffer_;
  }

  T &operator[](std::size_t index) noexcept
  {
    return static_cast<T *>(buffer_->contents())[index];
  }

  const T &operator[](std::size_t index) const noexcept
  {
    return static_cast<T *>(buffer_->contents())[index];
  }

  T &at(std::size_t index) noexcept
  {
    return static_cast<T *>(buffer_->contents())[index];
  }

  const T &at(std::size_t index) const noexcept
  {
    return static_cast<T *>(buffer_->contents())[index];
  }

  /**
   * Get buffer
   */
  [[nodiscard]] MTL::Buffer *GetBuffer() const noexcept
  {
    return buffer_->retain();
  }

 private:
  std::size_t buffer_length_;
  NS::SharedPtr<MTL::Buffer> buffer_;
};
} // namespace nagato
#endif //NAGATOLIB_METAL_METAL_BUFFER_HPP_
