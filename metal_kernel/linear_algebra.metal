#include <metal_stdlib>
#include <metal_compute>
#include <metal_atomic>  // atomic_float使用
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


/**
 * 1カーネルで配列の総和を求めるサンプル.
 *   - 「atomic_float」が使用できない環境を想定し、atomic_uintによるビットパニングで実装.
 *   - グループ内で部分和をまとめた後、グループ代表がatomicsでglobalSum[0]へ加算します.
 *
 * buffer(0): device const float*  (入力配列)
 * buffer(1): device uint*         (最終的な総和をビット形式で貯める領域/要素数=1)
 * buffer(2): constant uint*       (params[0] = 配列の長さ(arraySize), params[1] = 全スレッド数)
 * threadgroup(0): float*          (スレッドグループ内共有メモリ)
 */
kernel void sum_arrays_full(
  device const float* input       [[ buffer(0) ]],
  device atomic_uint* globalSum   [[ buffer(1) ]],
  constant uint*      params      [[ buffer(2) ]],
  threadgroup float*  sharedMem   [[ threadgroup(0) ]],
  uint tid                        [[ thread_position_in_threadgroup ]],
  uint groupId                    [[ threadgroup_position_in_grid ]],
  uint threadsPerThreadgroup      [[ threads_per_threadgroup ]]
)
{
  // params配列から取得
  uint arraySize   = params[0]; // 配列の長さ
  uint totalThreads= params[1]; // 全グリッド合計のスレッド数

  // グローバルIDを算出 (1次元dispatch想定)
  uint globalId = groupId * threadsPerThreadgroup + tid;

  // まず各スレッドが自分の担当分を足し合わせる
  float localSum = 0.0f;
  for (uint i = globalId; i < arraySize; i += totalThreads) {
    localSum += input[i];
  }

  // スレッドグループ内共有メモリへ書き込み
  sharedMem[tid] = localSum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // グループ内リダクション
  //  典型的な手法で半分ずつ足し合わせていく
  for (uint offset = threadsPerThreadgroup >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      sharedMem[tid] += sharedMem[tid + offset];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // グループ内スレッド0だけがatomicsで加算
  //  ここでは "ビットをuintに変換して" 加算している点に注意.
  if (tid == 0) {
    float groupSum = sharedMem[0];
    // float -> uint へビットパニング
    uint bitsToAdd = as_type<uint>(groupSum);

    atomic_fetch_add_explicit(globalSum, bitsToAdd, memory_order_relaxed);
  }
}