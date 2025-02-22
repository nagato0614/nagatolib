# nagatolib

よく使うクラス群

## 実装済みのもの
- Vector
- Matrix
- ThreadPool
- Random
- Metal (Metalのラッパー : macのみ有効)

## 実装したいもの
- メモリアロケータ

## テスト環境
- gtest

## NArray
多次元配列を取り扱う機能を提供するクラス
ゆくゆくはGPU (Metal) に対応させたいがまずはCPUでの実装を行う

| レイヤー            | Forward 入力形状                              | Forward 出力形状         | Backward 入力形状                           | Backward 出力形状         | パラメータ勾配形状         | パラメータ/内部変数形状                          |
|---------------------|-----------------------------------------------|--------------------------|---------------------------------------------|--------------------------|----------------------------|------------------------------------------------|
| 【Affine1 (W1, b1)】| (10, D) → (10, 1, D) ※入力を 3 次元に変換         | (10, 1, H)              | (10, 1, H) （上流からの勾配）                 | (10, 1, D)              | dW: (D, H)<br>db: (1, H)    | W1: (D, H)<br>b1: (1, H)<br>内部 x: (10, 1, D)  |
| 【ReLU】           | (10, 1, H)                                    | (10, 1, H)               | (10, 1, H)                                   | (10, 1, H)               | －                         | 内部保存 input: (10, 1, H)                        |
| 【Affine2 (W2, b2)】| (10, 1, H)                                    | (10, 1, O)              | (10, 1, O)                                   | (10, 1, H)              | dW: (H, O)<br>db: (1, O)    | W2: (H, O)<br>b2: (1, O)<br>内部 x: (10, 1, H)    |
| 【SoftmaxWithLoss】 | (10, 1, O) ※t: (10, 1, O) または (10, O)         | スカラー (loss)         | スカラー（通常 dout は 1 として与える）         | (10, 1, O)              | －                         | 内部 y: (10, 1, O)<br>t: (10, 1, O)<br>loss: スカラー |