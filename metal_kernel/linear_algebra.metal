#include <metal_stdlib>
#include <metal_compute>
#include <metal_atomic>

#include "../Metal/metal_common.hpp"
using namespace metal;

/**
 * 要素ごとの和を求める
 * @param inA
 * @param inB
 * @param result
 * @param index
 * @return
 */
kernel void add_arrays(device const float *inA         [[buffer(0)]],
                       device const float *inB         [[buffer(1)]],
                       device float       *result      [[buffer(2)]],
                       constant uint      &buffer_length,
                        uint              index        [[thread_position_in_grid]])
{
  // 1スレッドあたり DataSizePerThread 個の要素を処理する想定
  // 4ずつ進めて simd_float4 で読み書き
  for (uint i = 0; i < DataSizePerThread; i += 4) {
    uint dataIndex = index * DataSizePerThread + i;

    // 4要素まとめて処理可能かどうかをチェック
    if (dataIndex + 3 < buffer_length) {
      // メモリ上で simd_float4 としてキャストし、まとめて読み書き
      device const simd_float4* aPtr = reinterpret_cast<device const simd_float4*>(inA + dataIndex);
      device const simd_float4* bPtr = reinterpret_cast<device const simd_float4*>(inB + dataIndex);
      device       simd_float4* rPtr = reinterpret_cast<device       simd_float4*>(result + dataIndex);

      simd_float4 aVal = *aPtr;
      simd_float4 bVal = *bPtr;
      *rPtr = aVal + bVal;
    }
    else
    {
      // buffer_length が 4 の倍数でない場合の端数処理
      // 残りの要素を1つずつ処理する
      for (uint j = 0; j < 4; ++j) {
        uint idx = dataIndex + j;
        if (idx < buffer_length) {
          result[idx] = inA[idx] + inB[idx];
        }
      }
    }
  }
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
                       uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] / inB[index];
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
kernel void sum_arrays(
    device const float* input       [[buffer(0)]],
    device atomic_float* globalSum  [[buffer(1)]],
    constant uint*      params      [[buffer(2)]],
    threadgroup float*  sharedMem   [[threadgroup(0)]],
    uint tid                        [[thread_position_in_threadgroup]],
    uint groupId                    [[threadgroup_position_in_grid]],
    uint threadsPerThreadgroup      [[threads_per_threadgroup]]
)
{
    // params配列から取得
    uint arraySize    = params[0]; // 配列の長さ
    uint totalThreads = params[1]; // 全グリッド合計のスレッド数

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
    /*
     * リダクション（足し合わせの集約）の定番手法は、半分ずつ束ねていくという考え方です。
     * たとえば threadsPerThreadgroup = 8 だとして、それぞれに部分和が入っていると仮定すると:
     *
     * 共有メモリ:
     *          index:   0    1    2    3    4    5    6    7
     *             -------------------------------------
     * 初期値             v0   v1   v2   v3   v4   v5   v6   v7   (各tidの部分和)
     * 1回目 (offset=4) v0+v4 v1+v5 v2+v6 v3+v7   (4~7は上書きされるが使わない)
     * 2回目 (offset=2) (v0+v4)+(v2+v6)  (v1+v5)+(v3+v7)
     * 3回目 (offset=1) 全体の合計  ...
     *
     * このように、ループを回すたびに offset を半分にしながら、下位の要素が上位の要素を足し込んでいく
     * 最終的に sharedMem[0] にグループ全体の合計が残る
     **/
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

        // メモリ同期性の向上
        atomic_fetch_add_explicit(globalSum, groupSum, memory_order_relaxed);
    }
}

kernel void sqrt_arrays(device const float *in,
                       device float *out,
                       uint index [[thread_position_in_grid]])
{
    out[index] = sqrt(in[index]);
}

/**
 * Softmax 関数を計算するカーネル
 * cpu実装の方が早い
 */
kernel void softmax(
    device const float* input        [[buffer(0)]],
    device float*       output       [[buffer(1)]],
    device atomic_float* globalSum   [[buffer(2)]],
    constant uint&      arraySize    [[buffer(3)]],
    threadgroup float*  sharedMem    [[threadgroup(0)]],
    uint                tid          [[thread_position_in_threadgroup]],
    uint                groupId      [[threadgroup_position_in_grid]],
    uint                threadsPerThreadgroup [[threads_per_threadgroup]],
    uint                globalId     [[thread_position_in_grid]]
)
{
    if (globalId >= arraySize) return;

    // **ステップ 1: 最大値を求める**
    float localMax = input[globalId];
    localMax = simd_max(localMax);

    // **スレッドグループ内で最大値を求める**
    sharedMem[tid] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = threadsPerThreadgroup / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sharedMem[tid] = max(sharedMem[tid], sharedMem[tid + offset]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maxVal = sharedMem[0];

    // **ステップ 2: exp(x - max) を計算し、部分和を求める**
    float expValue = exp(input[globalId] - maxVal) + 1e-30f;  // アンダーフロー防止
    sharedMem[tid] = expValue;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float localSum = simd_sum(expValue);
    for (uint offset = threadsPerThreadgroup / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sharedMem[tid] += sharedMem[tid + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // **スレッドグループごとの部分和を 1 回だけ加算**
    if (tid == 0) {
        atomic_fetch_add_explicit(globalSum, sharedMem[0], memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // **ステップ 3: 合計値を取得して Softmax 計算**
    float sum = atomic_load_explicit(globalSum, memory_order_relaxed);
    float inv_sum = 1.0f / sum;
    output[globalId] = expValue * inv_sum;
}

kernel void sigmoid_array(
    device const float *in  [[buffer(0)]],
    device       float *out [[buffer(1)]],
    constant uint      &buffer_length,
    uint index              [[thread_position_in_grid]])
{
    float x = in[index];
    out[index] = 1.0f / (1.0f + exp(-x));
}

kernel void relu_arrays(
    device const float * in  [[buffer(0)]],
    device       float * out [[buffer(1)]],
    constant uint &buffer_length  [[buffer(2)]],
    uint index [[thread_position_in_grid]]
)
{
    // 1スレッドあたり DataSizePerThread 個の要素を処理
    // 4ずつ進めて simd_float4 で読み書きする
    for (uint i = 0; i < DataSizePerThread; i += 4) {
        uint dataIndex = index * DataSizePerThread + i;

        // 4要素まとめて処理可能かどうかをチェック
        if (dataIndex + 3 < buffer_length) {
            // メモリ上で simd_float4 としてキャストし、まとめて読み書き
            device const simd_float4* inPtr  = reinterpret_cast<device const simd_float4*>(in  + dataIndex);
            device       simd_float4* outPtr = reinterpret_cast<device       simd_float4*>(out + dataIndex);

            simd_float4 x = *inPtr;
            // ReLU(x) = max(0, x)
            *outPtr = max(x, simd_float4(0.0f));
        } else {
            // buffer_length が 4 の倍数でない場合の端数処理
            // 残りの要素を1つずつ処理
            for (uint j = 0; j < 4; ++j) {
                uint idx = dataIndex + j;
                if (idx < buffer_length) {
                    out[idx] = max(0.0f, in[idx]);
                }
            }
        }
    }
}
