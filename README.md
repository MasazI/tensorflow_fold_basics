# tensorflow_fold_basics
tensorflow_fold_basics は TensorFlow Fold で遊んだり、練習するためのレポジトリです。

## TensorFlow Flodの特徴

動的に変更するネットワークを描くための、TensorFlowベースのフレームワーク。
何故動的なネットワークが必要かというと、自然言語に代表されるシーケンスデータが、系列であるということ以上の構造をもつから。その構造は言語の場合には文法と呼ばれる。

この構造を木として表現し、木の深さごとにネットワークを準備して、積み上げると、構造を考慮したネットワークができあがる。
自然言語以外には、ソースコードのシンタックス木やWEBPageのDOM木などなど。

様々なサイズや構造をもったデータに対して、tensorflow foldは簡単に深層学習モデルを適用できる。かなりスピード・アップできる。
また、関数型を導入したため、ネットワークの結合やデータのFold処理をOperatorと関数を使ってスマートに記述できる。

## 遊び場

- sandbox.py

## 練習

- tiny_mlp.py


---

Copyright (c) 2017 Masahiro Imai
Released under the MIT license
