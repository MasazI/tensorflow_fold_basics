# tensorflow_fold_basics
tensorflow_fold_basics は TensorFlow Fold で遊んだり、練習するためのレポジトリです。

## TensorFlow Flodの特徴

動的に変更するネットワークを描くための、TensorFlowベースのフレームワーク。
何故動的なネットワークが必要かというと、自然言語に代表されるシーケンスデータが、系列であるということ以上の構造をもつから。その構造は言語の場合には文法と呼ばれる。
RNNは仕組み上、構造を保持することができない。 データが不安定な場合に、構造を考慮するネットワークを構築するには、データごとに異なるネットワークを動的に生成する必要がある。

このような構造を木として表現し、木の深さごとにネットワークを準備して、積み上げると、構造を考慮したネットワークをつくることができる。
自然言語以外には、ソースコードのシンタックス木やWEBPageのDOM木などなど。

様々なサイズや構造をもったデータに対して、tensorflow foldは簡単に深層学習モデルを適用できる。かなりスピード・アップできる。
また、関数型を導入したため、ネットワークの結合やデータのFold処理をOperatorと関数を使ってスマートに記述できる。

## 遊び場

- sandbox.py

ここでは TensorFlow fold の 簡単な neuralnetwork block と compiler による neuralnetwork の実行、実行して得られる値を使った loss定義 と 最適化の適用方法を実行する。


## 練習

- tiny_mlp.py

tiny_mlp では tensorflow fold の plan という考え方を学ぶ。plan に関数を登録して、セッションで実行する。planには必要な関数を設定する。plan を runすると設定された関数を組み合わせて学習や推論を行う。


---

Copyright (c) 2017 Masahiro Imai
Released under the MIT license
