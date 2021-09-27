# Array of Structure(AoS)とStructure of Array(SoA)の比較
GPU上で動かすアプリケーションの場合、直感的なAoSよりもSoAの方が実行速度が早いとされる。
このレポジトリでは複数の値(2~6個)を持つベクトル値の配列をAoS型、SoA型のメモリレイアウトで表現し、それに対していくつかの操作を試みた。

CUDA Cによる実装であるため、nvccを必要とする。

## Makefileの使い方
ベクトルの要素数に応じてmainN2.out, mainN3.out, mainN4.out, mainN5.out, mainN6.outの5つの実行ファイルを生成する。
make allで全ての実行ファイルをコンパイルする。
make runで全ての実行ファイルを実行する。
