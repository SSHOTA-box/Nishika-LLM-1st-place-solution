[English](README.md) | [日本語](README_JA.md)

# リヴァンプ × Nishika LLMコンペティション 1位解法
※日本語版が最新版です。より詳しい情報を知りたい場合は、日本語版をご覧いただくことをおすすめします。

チームSSS_labが実際に使用した学習・推論・アンサンブルコードです。
本コンペティションは、**株式会社リヴァンプ様**と**Nishika株式会社様**によって共同開催されました。商品画像や商品名などの商品情報から、商品の魅力を伝える自然で説得力のある日本語PR文を生成することが課題でした。単に商品スペックを列挙するのではなく、提供情報との整合性を保ちながら購買意欲につながる表現を作ることが求められました。

本リポジトリのコードが直接使用する入力は、商品IDと対応する1～2枚の商品画像です。

## 結果

| 項目 | 内容 |
| --- | --- |
| コンペ | 【学生限定】【リヴァンプ×Nishika LLMコンペティション】大手グローバル小売メーカーの商品PR文生成 |
| 順位 | 1位 / 48チーム |
| チーム | SSS_lab |
| 最終スコア | `0.639706` |
| タスク | 1～2枚の商品画像から100文字以内の日本語PR文を生成 |
| 評価指標 | 主催者提供のテキストエンコーダーによるコサイン類似度の平均 |

コンペの開催情報と公開条件は[コンペティション情報](docs/COMPETITION_JA.md)にまとめています。

- [株式会社リヴァンプ公式開催レポート](https://revamp.co.jp/interview/13/)

## 解法

![Qwen3-VL、LoRA、公式Sentence-BERT、埋め込み重心法を組み合わせた解法構成図](docs/assets/Solution.png)

1. Qwen3-VL-8B-Instructを4-bit NF4で読み込む
2. LoRAで商品PR文生成に追加学習する
3. Temperature 0.9と1.2で各100候補を生成する
4. 主催者提供のテキストエンコーダーで候補を埋め込む
5. 200候補の重心に最も近い文を選択する
6. 出力を100文字に整形する

比較実験と詳細な設定は[解法詳細](docs/SOLUTION_JA.md)を参照してください。

## リポジトリ構成

```text
.
├── code/
│   ├── train.py              # LoRA学習
│   ├── predict_v1.py         # Temperature 0.9、100候補
│   ├── predict_v2.py         # Temperature 1.2、100候補
│   └── marge.py              # 200候補の統合と最終出力
├── data/
│   ├── train.csv             # ID,label（架空のサンプルを同梱）
│   ├── test.csv              # ID（同梱しない）
│   └── sample_submission.csv # ID,target（同梱しない・コードでは未使用）
├── images/
│   ├── train/<ID>/*.{jpg,jpeg,png,bmp} # 架空のサンプル2枚を同梱
│   └── test/<ID>/*.{jpg,jpeg,png,bmp}  # 同梱しない
├── models/                   # ベースモデル、encoder、LoRA
├── outputs/                  # 推論結果
├── docs/                     # コンペ情報と解法資料
├── tests/
├── settings.json
├── Dockerfile
└── docker-compose.yml
```

元のコンペデータと画像は含めません。同梱するCSVと画像は配置形式を示すための架空のサンプルで、学習用データではありません。

## 必要環境

- DockerおよびDocker Compose
- NVIDIA GPU
- NVIDIA Container Toolkit

動作確認環境はNVIDIA RTX 6000 Ada（VRAM 48 GB）、RAM 64 GBです。

## セットアップ

リポジトリルートで実行します。

```bash
docker compose build
docker compose run --rm nishika-env python setup_dirs.py
docker compose run --rm nishika-env python setup_model_downloders.py
```

`setup_model_downloders.py`はQwen3-VL-8B-Instructを`models/Qwen3-VL-8B-Instruct/`へダウンロードします。

主催者提供の評価用エンコーダーは再配布せず、公開ダウンロード先も設定していません。正規の利用権限を持つ場合のみ、Hugging Face形式のmodel directory一式を次に配置してください。

```text
models/predicter_LLM/
```

動作確認には、Transformersの`AutoTokenizer`と`AutoModel`で読み込める一般的なテキストencoderでも代用できます。ただし、主催者提供encoderと同じ評価値は再現できません。

## モデル

| モデル | 配置先 | 公開状況 |
| --- | --- | --- |
| Qwen3-VL-8B-Instruct | `models/Qwen3-VL-8B-Instruct/` | setup scriptで取得 |
| 評価用encoder | `models/predicter_LLM/` | 主催者提供版は配布しない。一般的なencoderで代用可能 |
| 学習済みLoRA | `models/LoRA/LoRA_Qwen3-VL-8B-Instruct/` | 本リポジトリでは配布しない |

学習済みLoRA adapterは本リポジトリでは配布しません（公開は可能なため、必要な方はSShotaへ個別にご連絡ください）。

## 学習

```bash
docker compose run --rm nishika-env python code/train.py
```

主な設定はseed 42、末尾300件をvalidation、effective batch size 8、learning rate `1e-4`です。現在のコードは4 epochsです。

## 推論

`settings.json`の`LORA_ADAPTER`を使用するcheckpointに合わせてから実行します。コンペで使用した設定は`checkpoint-1599`です。

```bash
docker compose run --rm nishika-env python code/predict_v1.py
docker compose run --rm nishika-env python code/predict_v2.py
docker compose run --rm nishika-env python code/marge.py
```

最終ファイルは次に出力されます。

```text
outputs/LoRA_inference/final_submission/final_submission_ensemble_200.csv
```

## テスト

```bash
python -m unittest discover -s tests -v
```

## 公開範囲

公開するもの:

- 学習・推論・アンサンブルコード
- Docker環境と設定
- 解法および比較実験
- 架空のサンプルデータと画像

公開しないもの:

- 元のコンペデータと画像
- 主催者提供のテキストエンコーダー
- 学習済みLoRA adapter
- 生成した予測、提出物、ログ
- 個人情報を含む元のPDF・PowerPoint資料

公開範囲や解法について、気になる点や興味のある内容があれば、ぜひご連絡ください。質問も歓迎します。

## ライセンス

実装コードはSShotaが独自に作成し、[Apache License 2.0](LICENSE)で公開します。

コンペ固有の解法資料は、保存されたCompetition Rulesのコンペ終了後・非営利目的での公開条件に基づきます。適用範囲は[LICENSE_SCOPE.md](LICENSE_SCOPE.md)、外部依存は[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)を参照してください。

## Disclaimer

本リポジトリは株式会社リヴァンプおよびNishika株式会社の公式リポジトリではありません。企業名、コンペティション名、商標等の権利は各権利者に帰属します。
