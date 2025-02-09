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
                       uint               index        [[thread_position_in_grid]]
                       )
{
    // 1スレッドあたり DataSizePerThread 個の要素を処理する想定
    // 4ずつ進めて simd_float4 で読み書き
    for (uint i = 0; i < DataSizePerThread; i += 4) {
          uint dataIndex = index * DataSizePerThread + i;

          // 4要素まとめて処理可能かどうかをチェック
          if (dataIndex + 3 < buffer_length) {
              // メモリ上で simd_float4 としてキャストし、まとめて読み書き
              device const simd_float4* aPtr 
                = reinterpret_cast<device const simd_float4*>(inA + dataIndex);
              device const simd_float4* bPtr 
                = reinterpret_cast<device const simd_float4*>(inB + dataIndex);
              device       simd_float4* rPtr 
                = reinterpret_cast<device       simd_float4*>(result + dataIndex);

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
 * 要素ごとの和を求める. バッチ処理対応版
 * @param inA 入力配列1
 * @param inB 入力配列2
 * @param result 出力配列
 * @param batch_size バッチサイズ
 * @param index スレッドのインデックス
 * @return
 */
kernel void add_array_batch(device const float *inA             [[buffer(0)]],
                            device const float *inB             [[buffer(1)]],
                            device float       *result          [[buffer(2)]],
                            constant uint      &buffer_length   [[buffer(3)]],
                            constant uint      &batch_size      [[buffer(4)]],
                            ushort3            gid              [[thread_position_in_grid]]
                            )
{
    const uint thread_index = gid.x;
    const uint batch_index = gid.z;

    // 1スレッドあたり DataSizePerThread 個の要素を処理する想定
    // 4ずつ進めて simd_float4 で読み書き
    for (uint i = 0; i < DataSizePerThread; i += 4) {

        // 計算する要素のインデックスを算出. batch size を考慮
        uint dataIndex = batch_index * buffer_length + thread_index * DataSizePerThread + i;

        // 4要素まとめて処理可能かどうかをチェック
        if (dataIndex + 3 < buffer_length * batch_size) {
          // メモリ上で simd_float4 としてキャストし、まとめて読み書き
            device const simd_float4* aPtr = reinterpret_cast<device const simd_float4*>(inA +      dataIndex);
            device const simd_float4* bPtr = reinterpret_cast<device const simd_float4*>(inB +      dataIndex);
            device       simd_float4* rPtr = reinterpret_cast<device       simd_float4*>(result +   dataIndex);

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
 * 要素ごとの差を求める. バッチ処理対応版
 * @param inA 入力配列1
 * @param inB 入力配列2
 * @param result 出力配列
 * @param batch_size バッチサイズ
 * @param index スレッドのインデックス
 * @return
 */
kernel void sub_array_batch(device const float *inA,
                            device const float *inB,
                            device float *result,
                            constant uint &buffer_length,
                            constant uint &batch_size,
                            ushort3 gid [[thread_position_in_grid]])
{
    const uint thread_index = gid.x;
    const uint batch_index = gid.z;

    // 1スレッドあたり DataSizePerThread 個の要素を処理する想定
    // 4ずつ進めて simd_float4 で読み書き
    for (uint i = 0; i < DataSizePerThread; i += 4) {

        // 計算する要素のインデックスを算出. batch size を考慮
        uint dataIndex = batch_index * buffer_length + thread_index * DataSizePerThread + i;

        // 4要素まとめて処理可能かどうかをチェック
        if (dataIndex + 3 < buffer_length * batch_size) {
          // メモリ上で simd_float4 としてキャストし、まとめて読み書き
            device const simd_float4* aPtr = reinterpret_cast<device const simd_float4*>(inA +      dataIndex);
            device const simd_float4* bPtr = reinterpret_cast<device const simd_float4*>(inB +      dataIndex);
            device       simd_float4* rPtr = reinterpret_cast<device       simd_float4*>(result +   dataIndex);

          simd_float4 aVal = *aPtr;
          simd_float4 bVal = *bPtr;
          *rPtr = aVal - bVal;
        }
        else
        {
            // buffer_length が 4 の倍数でない場合の端数処理
            // 残りの要素を1つずつ処理する
            for (uint j = 0; j < 4; ++j) {
                uint idx = dataIndex + j;
                if (idx < buffer_length) {
                    result[idx] = inA[idx] - inB[idx];
                }
            }
        }
    }
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
 * 要素ごとの積を求める. バッチ処理対応版
 * @param inA 入力配列1
 * @param inB 入力配列2
 * @param result 出力配列
 * @param batch_size バッチサイズ
 * @param index スレッドのインデックス
 * @return
 */
kernel void mul_array_batch(device const float *inA,
                            device const float *inB,
                            device float *result,
                            constant uint &buffer_length,
                            constant uint &batch_size,
                            ushort3 gid [[thread_position_in_grid]])
{
    const uint thread_index = gid.x;
    const uint batch_index = gid.z;

    // 1スレッドあたり DataSizePerThread 個の要素を処理する想定
    // 4ずつ進めて simd_float4 で読み書き
    for (uint i = 0; i < DataSizePerThread; i += 4) {

        // 計算する要素のインデックスを算出. batch size を考慮
        uint dataIndex = batch_index * buffer_length + thread_index * DataSizePerThread + i;

        // 4要素まとめて処理可能かどうかをチェック
        if (dataIndex + 3 < buffer_length * batch_size) {
          // メモリ上で simd_float4 としてキャストし、まとめて読み書き
            device const simd_float4* aPtr = reinterpret_cast<device const simd_float4*>(inA +      dataIndex);
            device const simd_float4* bPtr = reinterpret_cast<device const simd_float4*>(inB +      dataIndex);
            device       simd_float4* rPtr = reinterpret_cast<device       simd_float4*>(result +   dataIndex);

          simd_float4 aVal = *aPtr;
          simd_float4 bVal = *bPtr;
          *rPtr = aVal * bVal;
        }
        else
        {
            // buffer_length が 4 の倍数でない場合の端数処理
            // 残りの要素を1つずつ処理する
            for (uint j = 0; j < 4; ++j) {
                uint idx = dataIndex + j;
                if (idx < buffer_length) {
                    result[idx] = inA[idx] * inB[idx];
                }
            }
        }
    }
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
 * 要素ごとの商を求める. バッチ処理対応版
 * @param inA 入力配列1
 * @param inB 入力配列2
 * @param result 出力配列
 * @param batch_size バッチサイズ
 * @param index スレッドのインデックス
 * @return
 */
kernel void div_array_batch(device const float *inA,
                            device const float *inB,
                            device float *result,
                            constant uint &buffer_length,
                            constant uint &batch_size,
                            ushort3 gid [[thread_position_in_grid]])
{
    const uint thread_index = gid.x;
    const uint batch_index = gid.z;

    // 1スレッドあたり DataSizePerThread 個の要素を処理する想定
    // 4ずつ進めて simd_float4 で読み書き
    for (uint i = 0; i < DataSizePerThread; i += 4) {

        // 計算する要素のインデックスを算出. batch size を考慮
        uint dataIndex = batch_index * buffer_length + thread_index * DataSizePerThread + i;

        // 4要素まとめて処理可能かどうかをチェック
        if (dataIndex + 3 < buffer_length * batch_size) {
          // メモリ上で simd_float4 としてキャストし、まとめて読み書き
            device const simd_float4* aPtr = reinterpret_cast<device const simd_float4*>(inA +      dataIndex);
            device const simd_float4* bPtr = reinterpret_cast<device const simd_float4*>(inB +      dataIndex);
            device       simd_float4* rPtr = reinterpret_cast<device       simd_float4*>(result +   dataIndex);

          simd_float4 aVal = *aPtr;
          simd_float4 bVal = *bPtr;
          *rPtr = aVal / bVal;
        }
        else
        {
            // buffer_length が 4 の倍数でない場合の端数処理
            // 残りの要素を1つずつ処理する
            for (uint j = 0; j < 4; ++j) {
                uint idx = dataIndex + j;
                if (idx < buffer_length) {
                    result[idx] = inA[idx] / inB[idx];
                }
            }
        }
    }
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
     *          index:   0     1    2    3    4    5    6    7
     *             -------------------------------------
     * 初期値             v0   v1   v2   v3   v4   v5   v6   v7   (各tidの部分和)
     * 1回目 (offset=4)  v0+v4 v1+v5 v2+v6 v3+v7   (4~7は上書きされるが使わない)
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

kernel void sqrt_array_batch(device const float *in,
                             device float *out,
                             constant uint &buffer_length,
                             constant uint &batch_size,
                             ushort3 gid [[thread_position_in_grid]])
{
    const uint thread_index = gid.x;
    const uint batch_index = gid.z;

    for (uint i = 0; i < DataSizePerThread; i += 4) {
        uint dataIndex = batch_index * buffer_length + thread_index * DataSizePerThread + i;
        
        if (dataIndex + 3 < buffer_length * batch_size) {
            device const simd_float4* inPtr = reinterpret_cast<device const simd_float4*>(in + dataIndex);
            device       simd_float4* outPtr = reinterpret_cast<device       simd_float4*>(out + dataIndex);

            simd_float4 x = *inPtr;
            *outPtr = sqrt(x);
        }
        else
        {
            for (uint j = 0; j < 4; ++j) {
                uint idx = dataIndex + j;
                if (idx < buffer_length) {
                    out[idx] = sqrt(in[idx]);
                }
            }
        }
    }
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
    // 修正: スレッドグループ内での最大値を正しく求める
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

    // 修正: 部分和の計算を正しく行う
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

kernel void relu_array(
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

/**
 * A: NxM 行列 (row-major)
 * B: MxL 行列 (row-major)
 * C: NxL 行列 (row-major)
 * N, M, L: 行列サイズ
 * gid: グローバル座標 (x=列, y=行)
 */
kernel void matmul_array(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float       *C [[buffer(2)]],
    constant uint &N       [[buffer(3)]],
    constant uint &M       [[buffer(4)]],
    constant uint &L       [[buffer(5)]],
    
    ushort2 gid [[thread_position_in_grid]]
)
{
    // グローバル座標から 行(row) と 列(col) を取得
    // gid.x : 列 (0..L-1), gid.y : 行 (0..N-1)
    uint col = gid.x;
    uint row = gid.y;

    // 範囲外なら何もしない
    if (col >= L || row >= N) {
        return;
    }

    // 素朴に行列積を計算
    // C[row, col] = Σ (A[row, k] * B[k, col]) for k in [0..M-1]
    float sum = 0.0f;
    for (uint k = 0; k < M; k++) {
        sum += A[row * M + k] * B[k * L + col];
    }

    // 結果をCに書き込む
    C[row * L + col] = sum;
}

/**
 * A: NxM 行列 (row-major)
 * B: MxL 行列 (row-major)
 * C: NxL 行列 (row-major)
 * batch_size: バッチサイズ
 * N, M, L: 行列サイズ
 * gid: グローバル座標 (x=列, y=行)
 */
kernel void matmul_array_batch(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float       *C [[buffer(2)]],
    constant uint &N       [[buffer(3)]],
    constant uint &M       [[buffer(4)]],
    constant uint &L       [[buffer(5)]],
    constant uint &batch_size [[buffer(6)]],
    ushort3 gid [[thread_position_in_grid]]
)
{
    uint col = gid.x;
    uint row = gid.y;
    uint batch_index = gid.z;

    if (col >= L || row >= N || batch_index >= batch_size) {
        return;
    }

    float sum = 0.0f;
    for (uint k = 0; k < M; k++) {
        sum += A[batch_index * N * M + row * M + k] * B[batch_index * M * L + k * L + col];
    }

    C[batch_index * N * L + row * L + col] = sum;
}


/**
 * ベクトルの内積 (dot product) を求めるカーネル
 *
 * A      : 入力ベクトル1
 * B      : 入力ベクトル2
 * globalSum : 各スレッドグループで計算された部分和を
 *             グローバルな内積として加算するための領域 (atomic_float)
 * length : ベクトルの要素数
 *
 * 各スレッドは、自身のglobalIdが有効な場合、A[globalId] * B[globalId] を計算し、
 * スレッドグループ内でリダクションを行った後、グループ代表がグローバルな内積に加算します.
 */
kernel void dot_product(
    device const float *A                         [[buffer(0)]],
    device const float *B                         [[buffer(1)]],
    device atomic_float *globalSum                [[buffer(2)]],
    constant uint &length                         [[buffer(3)]],
    threadgroup float* sharedMem                  [[threadgroup(0)]],
    uint tid                                      [[thread_position_in_threadgroup]],
    uint groupId                                  [[threadgroup_position_in_grid]],
    uint threadsPerThreadgroup                    [[threads_per_threadgroup]],
    uint globalId                                 [[thread_position_in_grid]]
)
{
    float partialSum = 0.0f;
    // 各スレッドが担当する要素の内積を計算 (グローバルIDが範囲内の場合)
    if (globalId < length) {
        partialSum = A[globalId] * B[globalId];
    }
    
    // 共有メモリに部分和を保存
    sharedMem[tid] = partialSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // スレッドグループ内でのリダクション処理
    for (uint offset = threadsPerThreadgroup >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sharedMem[tid] += sharedMem[tid + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // グループ内リダクション結果(部分和)をグローバルな内積に加算
    if (tid == 0) {
        atomic_fetch_add_explicit(globalSum, sharedMem[0], memory_order_relaxed);
    }
}
