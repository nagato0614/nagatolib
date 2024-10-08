#include <metal_stdlib>
#include <metal_compute>

#include "../Metal/metal_common.hpp"
using namespace metal;

/**
 * 要素ごとの和を求める
 * TODO : スレッドごとにいくつかの要素を処理するようにする
 * @param inA
 * @param inB
 * @param result
 * @param index
 * @return
 */
kernel void add_arrays(device const float *inA,
                       device const float *inB,
                       device float *result,
                       uint index [[thread_position_in_grid]])
{
  result[index] = inA[index] + inB[index];
}

/**
 * 要素ごとの差を求める
 * TODO : スレッドごとにいくつかの要素を処理するようにする
 * @param inA
 * @param inB
 * @param result
 * @param index
 * @return
 */
kernel void sub_arrays(device const float *inA,
                       device const float *inB,
                       device float *result,
                       uint index [[thread_position_in_grid]])
{
  result[index] = inA[index] - inB[index];
}

/**
 * 要素ごとの積を求める
 * TODO : スレッドごとにいくつかの要素を処理するようにする
 * @param inA
 * @param inB
 * @param result
 * @param index
 * @return
 */
kernel void mul_arrays(device const float *inA,
                       device const float *inB,
                       device float *result,
                       uint index [[thread_position_in_grid]])
{
  result[index] = inA[index] * inB[index];
}

/**
 * 要素ごとの商を求める
 * 分母が0の場合は考慮されていないため注意
 * TODO : スレッドごとにいくつかの要素を処理するようにする
 * @param inA
 * @param inB
 * @param result
 * @param index
 * @return
 */
kernel void div_arrays(device const float *inA,
                       device const float *inB,
                       device float *result,
                       uint index [[thread_position_in_grid]]
)
{
  result[index] = inA[index] / inB[index];
}

// グループごとに総和を求める
kernel void sum_arrays(
  device const float *input [[buffer(0)]], // 入力配列
  device float *output [[buffer(1)]],      // 出力配列 (総和)
  device uint *array_size [[buffer(2)]],            // 配列の長さ
  threadgroup float * buf [[threadgroup(0)]], // スレッドグループの共有メモリ
  uint index [[thread_position_in_grid]] // スレッドのインデックス
)
{
  // 1 スレッドあたり処理するデータ数
  uint process_size = data_size_per_thread;

  uint start = index * process_size;
  uint end = start + process_size < *array_size ? start + process_size : *array_size;

  // スレッドごとに総和を求める
  float sum = 0;
  for (uint i = start; i < end; i++)
  {
    sum += input[i];
  }
  output[index] = sum;
}

