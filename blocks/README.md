# TensorFlow Fold
tensorflow foldとは何か、どうやって使えるか、主要な特徴についてまとめる。

# tensorflow foldとは何か

動的に変更するネットワークを描くための、TensorFlowベースのフレームワーク。
何故動的なネットワークが必要かというと、自然言語に代表されるシーケンスデータが、
系列であるということ以上の構造をもつから。言語の場合には文法と呼ぶ。

この構造を木として表現し、木の深さごとにネットワークを準備して、積み上げると、構造を
考慮したネットワークができあがる。
自然言語以外には、ソースコードのシンタックス木やWEBPageのDOM木などなど。

様々なサイズや構造をもったデータに対して、tensorflow foldは簡単に深層学習モデルを適用できる。

さらに、かなりスピード・アップできる。
また、関数型の考え方を導入したため、よりスマートな記述ができる。

# どうやって使えるか

まず依存関係だが、tensorflow 1.0rc のインストールが必須。古いバージョンだとうまくいかない。

基本的な要素はblocksと呼ぶ。blocksにはいくつかの種類がある。

- Scalar
- Vector
- Record

Recordは素のTensorFlowより少しリッチで、タプルをが使える。
また、Operator >> を使って、関数を実行できる。

すなわち、Recordブロックに対して、Concat関数を実行したければ、以下のように記述すればよい。
record_block >> td.Concat()

これは関数型の記述であり、データに対する処理をスマートに記述できる。

これだけではなく、Foldの協力なデータ型 Mapがある。

- Map

Mapは不定なシーケンスデータの型であり、TensorFlowには存在しない。しかし、TensorFlowでもRNNやseq-to-seqが使えた。Foldの狙いはなんなのか？ジャグ配列（配列の要素も配列）なども扱えるのだ。

各Mapにももちろん 関数 と Operator >> は適用できる。
他にも関数として以下がある。

- Fold
- Reduce

Map型にはシーケンスを流すことができる。Map型のデータに対してFoldやReduceするための関数を渡すことができ、適用は Operator >> で記述する。

この流れで、ニューラルネットワークのレイヤーをデータに対して適用できる。動的バッチングも可能である。

関数型で学習済み結果を得ると、識別実行も非常にスマートになる。
feedするデータを生成するbuild_feed_dictもシンプルに利用できる。関数の出力を変えれば画像データも同じ方法で使うことができる。

# Blocks

blocks tutorial
https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/blocks.md

## Flodモデルへの入力

Pythonオブジェクトのミニバッチを使うことができる。プロトコルバッファ、JSON、XML、またはカスタムパーサででシリアラウズした木構造のデータ。

## Foldモデルの出力

TensorFlowテンソルの集合。通常は、この出力をloss関数や最適化関数に接続する。


## Foldの計算

入力データによって、効率的に実行する方法によってスケジューリングする。
例えば、木の各ノードが総結合ネットワークを使ってベクトルを出力する場合、Foldは単にトラバースしてベクトルを行列化して束ねて並列実行する代わりに、同じ深さのノードをマージして、もっとも大きい、効率化した行列の積演算にし、演算で出力される行列をスプリットして出力ベクトルにする。


## 階層型LSTMの例
****
次のコードはFoldでは簡単に、TensorFlowで実装するのは難しい階層型LSTMである。

```
# Create RNN cells using the TensorFlow RNN library
char_cell = td.ScopedLayer(tf.contrib.rnn.BasicLSTMCell(num_units=16), 'char_cell')
word_cell = td.ScopedLayer(tf.contrib.rnn.BasicLSTMCell(num_units=32), 'word_cell')

# character LSTM converts a string to a word vector
char_lstm = (td.InputTransform(lambda s: [ord(c) for c in s]) >>
td.Map(td.Scalar('int32') >>
td.Function(td.Embedding(128, 8))) >>
td.RNN(char_cell))
# word LSTM converts a sequence of word vectors to a sentence vector.
word_lstm = td.Map(char_lstm >> td.GetItem(1)) >> td.RNN(word_cell)
```
**階層型LSTMの入力：文字列のリスト、各文字列は単語**
**階層型LSTMの出力：文脈ベクトル**

2つのネストしたLSTMを使って、実行する。

**1つめのLSTM**=文字（キャラクター）LSTM：インプットとして文字列、出力は単語ベクトル（この処理は文字列をイント型の1つのリストに変換する。各イント型の数値は、埋め込みテーブルをルックアップしている）
**2つめのLSTM**=単語（word）LSTM：文字（キャラクター）LSTMを単語のシーケンス（各文字）にマッピング（各文字ごとにchar_lstmを適用してItemを1つ取得する）して、単語ベクトルのシーケンスを取得し、単語ベクトルを第２の大きなLSTMで処理して、文章ベクトルを生成する。


## 基本的コンセプト

Foldモデルの基本のコンポーネントはtd.Blockである。
td.Blockは基本的に関数であり、入力としてあるオブジェクトを受け取り、出力として別のオブジェクトを生成する。そのオブジェクトはテンソルであってもよし、タプル、リスト、Pythonの辞書、それらの組み合わせでもよい。

Blocksは階層的なツリーで編成されている、プログラミング言語の表現のように。
大きくて複雑なblocksはより小さくて、シンプルなblocksで構成されている。
Blocksは1つのツリーを構成していなければならず、ツリー内で固有の位置をもたなければならない。型チェックとコンパイルのステップは木のプロパティに依存する。


## プリミティブブロック（Primitive blocks）
****
PrimitiveBlockはブロック階層の葉である。テンソルについての基本的な計算を担当する。
基本的な計算

- td.Scalar()：PythonスカラーをTensorに変換
- td.Vector(shape)：Pythonリストを与えられたShapeのTensorに変換
- td.FromTensor：TensorFlowテンソルをtdのblockでラップする。引数無し関数と同様に、FromTensorブロックは入力を受付ず、単に出力と対応するテンソルをつくるだけ。numpyのテンソルで使うこともできる。


## 関数とレイヤー

td.Functionブロックは、TensorFlowのオペレーションをブロックにラップする。入力としてテンソルを受け取り、テンソルを出力する。Functionsは複数の引数をとる、でなければ、複数の結果を生成し、入出力としてテンソルのタプルを処理する。

例えば、td.Function(tf.add) とすると、2つのテンソルのタプルを受け取り、1つのテンソル（和）を出力として生成する。

Fuctionブロックは、ニューラルネットワークのレイヤーと組み合わせて仕様して、総結合ネットワークや埋め込みなどの計算を実行できる。
td.Layerは呼び出し可能なPythonオブジェクトで、そのレイヤーの、異なるインスタンス間で重みを共有する。

以下の例、ffnet は2層のフィードフォーワードネットワークで、重みを共有している。

# fclayer defines the weights for a fully-connected layer with 1024 hidden units.
fclayer = td.FC(1024)
# Each call to Function(fclayer) creates a fully-connected layer,
# all of which share the weights provided by the fclayer object.
ffnet = td.Function(fclayer) >>  td.Function(fclayer) >> td.Function(fclayer)

オペレータ >> については、Blockを使った記述の説明を参照。


## Pythonオペレータ

td.InputTransform ブロックは任意のPython関数をBlockにラップする。
Pythonオブジェクトを入力として受け取り、Pythonオブジェクトを出力として返す。
例えば、次のブロックは、Python文字列を浮動小数点数のリストに変換する。
```
td.InputTransform(lambda s: [ord(c)/255.0 for c in s])
```
その名前が示すように、InputTransformは、Pythonの入力データを前処理してTensorFlowに渡すために使用される。データがTensorFlowのパイプラインの部分に到達すると、Python codeはもはや実行できない、すると Foldeは型エラーを生成する。


## ブロックの構成

ブロックは、より複雑なブロックを生成するために、いろいろな方法で他のブロックと構成できる。
**Wiring blocks together（ブロックをまとめてつなげる）**
最も簡単な構成は、オペレータ >> を使って、あるブロックの出力を、別のブロックの入力につなげるものである。シンタックス f >> g は関数の構造を示す。入力をfに送り、fの出力をgに送り、gの出力を返す、新しいブロックが作成される。
例えば以下のように。
```
mnist_model = (td.InputTransform(lambda s: [ord(c) / 255.0 for c in s]) >>
td.Vector(784) >>             # convert python list to tensor
td.Function(td.FC(100)) >>    # layer 1, 100 hidden units
td.Function(td.FC(100)))      # layer 2, 100 hidden units
```
mnist_modelは新しいBlockである。

**Dealing with sequences（シーケンス処理）**
Foldはシーケンスデータを、mapやfoldのような高次の関数に類似したブロックを使って処理する。
Sequencesは任意の長さになり、事例ごとに長さが様々になる。シーケンスを予め定義した長さに切り捨てたり、埋め込んだりする必要はない。

- td.Map(f)：シーケンスを入力としてとり、全ての要素にブロックfを適用して、シーケンスの出力として生成する。
- td.Fold(f, z)：シーケンスを入力としてとり、ブロックzの出力を初期要素として使って、fを実行する。
- td.RNN(c)：MapとFoldを組み合わせた、1つのリカレントニューラルネットワークである。初期状態と入力シーケンスをとり、rnn-cell c を、前の状態と入力から、新しい状態と出力を生成するために使う。最終状態を出力シーケンスを返す。
- td.Reduce(f)：シーケンスを入力としてとり、fをペアの要素に実行することで単一の値に削減する。基本的に、fでバイナリ木を実行する。
- td.Zip()：シーケンスのタプルを入力としてとり、タプルのシーケンスを出力として生成する。（pythonのzipと似ている）
- td.Broadcast(a)：ブロックのaの出力を取り出し、それを無限の反復シーケンスに変換する。通常、zipとmapと一緒に使われ、aを使用する関数で、シーケンスの各要素を処理する。

以下がこれらを使用した例である。
```
# Convert a python list of scalars to a sequence of tensors, and take the
# absolute value of each one.
abs = td.Map(td.Scalar() >> td.Function(tf.abs))

# Compute the sum of a sequence, processing elements in order.
sum = td.Fold(td.Function(tf.add), td.FromTensor(tf.zeros(shape)))

# Compute the sum of a sequence, in parallel.
sum = td.Reduce(td.Function(tf.add))

# Convert a string to a vector with a character RNN, using Map and Fold
char_rnn = (td.InputTransform(lambda s: [ord(c) for c in s]) >>
# Embed each character using an embedding table of size 128 x 16
td.Map(td.Scalar('int32') >>
td.Function(td.Embedding(128, 16))) >>
# Fold over the sequence of embedded characters,
# producing an output vector of length 64.
td.Fold(td.Concat() >> td.Function(td.FC(64)),
td.FromTensor(tf.zeros(64))))
```
FoldブロックとRNNブロックは、LSTMのようなシーケンスモデルをデータのリストに適用するために使用できる。1つの注意点は、非常に長いシーケンス（100 elementsを超えるような）に対するバックプロパゲーションでは勾配が消失する問題が現れることである。
Reduceの深さがシーケンスの長さの対数であるので、Foldを使うよりもReduceを使うほうが好ましいかもしれない。
さらに、長いシーケンスを処理しているときにTensorFlow自体がメモリ不足になることがある、勾配を計算するのに、すべての中間結果を保持しておく必要があるからである。よって、Foldはどんな長さのシーケンスでも処理できるのだが、長いシーケンスを、td.InputTransformを使って管理可能な長さまで切り捨てることが望ましい場合もある。

**Dealing with records（レコード処理）**
レコードは、それぞれがPython 辞書やプロトコルバッファのような異なる型をもつ名前付きのフィールドセットである。ある td.Recordブロックは入力として1つのrecordをとり、各フィールドに子ブロックを適用し、結果をタプルに結合して出力として生成する。その出力タプルは出力ベクトルを得るためにtd.Concat()に渡すことができる。

例えば、次のブロックは3つのフィールドのレコードを128次元のベクトルに変換する。これは、埋め込みテーブルを使ってidフィールドをベクトルに変換し、名前フィールドを上で定義した文字RNNを使用して実行し、ロケーションフィールドをそのまま使って、3つ全部の結果を結合して、総結合レイヤーに渡すものである。
```
# Takes as input records of the form:
# {'id':       some_id_number,
#  'name':     some_string,
#  'location': (x,y)
# }
rec = (td.Record([('id', td.Scalar('int32') >>
td.Function(td.Embedding(num_ids, embed_len))),
('name', char_rnn),
('location', td.Vector(2))]) >>
td.Concat() >> td.Function(td.FC(128)))
```
総結合レイヤーは128個の隠れユニットをもち、よって長さ128のベクトルを出力する。
それはrecがコンパイルされたときの入力 (embed_len + 64 + 2)のサイズを推測している。

**Wiring things together, in more complicated ways（より複雑な方法を使った結合）**
オペレータ>>を使ったシンプルな構成は、標準のUnixパイプに似ている。通常、大抵の場合は、特に、入力データ構造を公団するRecordとFoldのようなブロックを使って、組み合わせると十分である。

しかしながら、いくつかのモデルでは、より複雑な結合を行う必要があるだろう。
td.Composition ブロックはそのブロックの子供の入力と出力を任意のDAGにまとめてつなげることができる。

次のコードはLSTMセルをブロックとして定義している。これは、前に説明したようなRNNブロックでの使用に適している。lstm_cell.scope()内で定義されたすべてのブロックはlstm_cellの子供になる。
**b.reads(…)メソッド**は、他のブロックまたはブロックのタプルの出力を、**bの入力に結びつける**。
bの出力がタプルの場合は、b[i]でタプルの個々の要素を選択できる。
下の場合は、lstm_cell.input[1]（状態ベクトル）を受け取って、in_stateとし、lsmt_cell.input[0] でcellへの入力、in_state[1] 
```
# The input to lstm_cell is (input_vec, (previous_cell_state, previous_output_vec))
# The output of lstm_cell is (output_vec, (next_cell_state, output_vec))
lstm_cell = td.Composition()
with lstm_cell.scope():
in_state = td.Identity().reads(lstm_cell.input[1])
bx = td.Concat().reads(lstm_cell.input[0], in_state[1])     # inputs to gates
bi = td.Function(td.FC(num_hidden, tf.nn.sigmoid)).reads(bx)  # input gate
bf = td.Function(td.FC(num_hidden, tf.nn.sigmoid)).reads(bx)  # forget gate
bo = td.Function(td.FC(num_hidden, tf.nn.sigmoid)).reads(bx)  # output gate
bg = td.Function(td.FC(num_hidden, tf.nn.tanh)).reads(bx)     # modulation
bc = td.Function(lambda c,i,f,g: c*f + i*g).reads(in_state[0], bi, bf, bg)
by = td.Function(lambda c,o: tf.tanh(c) * o).reads(bc, bo)    # final output (output vec)
out_state = td.Identity().reads(bc, by)   # make a tuple of (bc, by) (next_cell_state, output_vec)
lstm_cell.output.reads(by, out_state) # (output_vec, (next_cell_state, output_vec))
```
この定義は、説明するために提供されている。セルの本体は対応するTensorFlow操作をラップする関数ブロックのみに依存しているので、LSTMセルをシンプルなレイヤーで直接実装するほうが、td.Composition にするより効率的である。

**Recursion and forward declarations（再帰と前方宣言）**
ツリー再帰型ニューラルネットワークを実装するには、再帰的ブロックの定義が必要である。
ブロックの型は、最初にtd.ForwardDeclarationで宣言される。その後、ブロック自体は順方向の定義を使って、再帰の参照を定義される。
td.ForwardDeclaration.resolve_to は再帰の定義を、順方向の宣言を使って縛ることができる。
例えば、これは算術式を評価するブロックで、TensorFlowを使ったものである。
```
# the expr block processes objects of the form:
# expr_type ::=  {'op': 'lit', 'val': <float>}
#             |  {'op': 'add', 'left': <expr_type>, 'right': <expr_type>}
expr_fwd = td.ForwardDeclaration(pvt.PyObjectType(), pvt.Scalar())
lit_case = td.GetItem('val') >> td.Scalar()
add_case = (td.Record({'left': expr_fwd(), 'right': expr_fwd()}) >>
td.Function(tf.add))
expr = td.OneOf(lambda x: x['op'], {'lit': lit_case, 'add': add_case})
expr_fwd.resolve_to(expr)

expr_fwdは宣言であり、ブロックではない。expr_fwd()を呼び出すたびに、宣言を参照するブロックが作成される。

```
