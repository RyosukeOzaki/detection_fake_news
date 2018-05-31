## Automatic Detection of Fake News
与えられたステートメントがフェイクニュースであるか自動で判定するための教師ありデータセットと学習に用いたニューラネットモデルの公開。[検索エンジンによる上位検索ページを情報源とする
フェイクニュース自動検出のためのデータセット作成(2018言語処理学会)](http://anlp.jp/proceedings/annual_meeting/2018/pdf_dir/E5-4.pdf)

## 環境
 [Word by Word Attention(WBW)](https://arxiv.org/pdf/1509.06664.pdf)<br>
・Python2.7 <br>
・Tensorflow-gpu1.2.0<br>
・Word2vec(GoogleNews-vectors-negative300.bin)<br>
[Three Level Hierarchical Attention Network(3HAN)](https://www.researchgate.net/publication/319306895_3HAN_A_Deep_Neural_Network_for_Fake_News_Detection)<br>
・Python3<br>
・Tensorflow-gpu1.2.0<br>
・Glove(glove.6B.100d.txt)
## データセット概要
<img width="997" alt="2018-05-31 13 44 55" src="https://user-images.githubusercontent.com/38748341/40763168-cbd239a0-64de-11e8-8ef7-d1003ca9897c.png">

1:Statementは[PolitiFactサイト](http://www.politifact.com/)から抽出したセンテンス<br>

2:Knowledge BaseをWebとしてStatementをクエリとしGoogle検索した検索結果(上位20)の記事のタイトル、本文、URLを抽出<br>

3:Statementと抽出した記事の本文の関係をTrue,False,Can not judge,Unrelatedの4値に4人のアノテーターの多数決によってアノテーションした正解ラベル<br>

True--賛成または同じ主張<br>
False--反対または偽だと主張<br>
Can not judge--関連記事であるが真偽判断は不可<br>
Unrelated--無関係な記事<br>

データセットの例<br>

|    |    |
|-------|------|
|  Statement  |  The Earth is not warming.  |
|  PolitiFactSite  |  http://www.politifact.com/texas/statements/2013/dec/13/barry-smitherman/scientific-consensus-remains-planet-warming/  |
|  PolitiFactLabel  |  False  |
|  Majority  |  False  |
|  Annotator1  |  False |
|  Annotator2  | False  |
|  Annotator3  |  True |
|  Annotator4  |  False |
|  Evidence   | Global warming is still happening. |
|  Title  | Global cooling Is global warming still happening?  |
|  Article   | To say we’re currently experiencing global cooling overlooks one simple physical reality............................ |

Majorityは４人のアノテーションの多数決できまる。<br>
また、Evidenceはアノテーションする際に記事のどの部分が根拠となりその判断をしたのかを示すため、記事から根拠部分を抽出した文である。

## モデル
### ・WBW
[REASONING ABOUT ENTAILMENT WITHNEURAL ATTENTION](https://arxiv.org/pdf/1509.06664.pdf)<br>
run

```
python main.py --train
```

### ・3HAN
[3HAN: A Deep Neural Network for Fake News Detection](https://www.researchgate.net/publication/319306895_3HAN_A_Deep_Neural_Network_for_Fake_News_Detection)
run

```
python train.py
```
<br><br>
**<u>データセットをtest用とtrain用で分割して、WBW,3HANが読み込める用にデータセットの前処理が必要</u>**
