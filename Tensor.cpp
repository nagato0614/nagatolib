Tensor Tensor::FromArray(const std::vector<float>& data, const std::vector<std::size_t>& shape) {
    Tensor t;
    t.shape_ = shape;
    std::size_t ndims = shape.size();
    t.strides_.resize(ndims);
    if (ndims > 0) {
        // 最後の次元のストライドは 1
        t.strides_[ndims - 1] = 1;
        // 後ろから順にストライドを計算: strides[i] = strides[i+1] * shape[i+1]
        for (int i = ndims - 2; i >= 0; --i) {
            t.strides_[i] = t.strides_[i + 1] * shape[i + 1];
        }
    }
    t.storage_ = data;
    return t;
} 