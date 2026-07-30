# SDR FM Radio Receiver - 技術仕様書

RTL-SDR を用いた FM ステレオ放送受信機の技術仕様書です。本書は現行
実装（2026-07 時点）を反映します。

## 目次

1. [システム概要](#1-システム概要)
2. [システムアーキテクチャ](#2-システムアーキテクチャ)
3. [モジュール詳細仕様](#3-モジュール詳細仕様)
4. [データフロー](#4-データフロー)
5. [信号処理パイプライン](#5-信号処理パイプライン)
6. [設定パラメータ](#6-設定パラメータ)
7. [インターフェース仕様](#7-インターフェース仕様)
8. [使用方法](#8-使用方法)
9. [録音ファイル](#9-録音ファイル)
10. [テストと CI](#10-テストと-ci)

---

## 1. システム概要

### 1.1 プロジェクト概要

RTL-SDR デバイスから取得した IQ サンプルを FM 復調し、ステレオ音声を
リアルタイム再生・録音するアプリケーションです。復調・ステレオ分離・
ノイズリダクションはすべて Python（NumPy / SciPy / Numba）で実装され
ています。

### 1.2 主要機能

- RTL-SDR による FM 放送受信
- arctan discriminator ベースの FM 復調（レガシー PLL 経路も選択可）
- ステレオ / モノラル出力（パイロット SNR 連動の適応ブレンド）
- Mid/Side スペクトル雑音抑制（side チャネル専用 DD-Wiener STFT）
- ソフトウェア自動ゲイン制御（AGC）
- リアルタイム音声再生・録音（音声 WAV / 生 IQ WAV）
- 4 GiB を越える長時間録音の自動ローテーションとメタデータ記録
- 合成 IQ / 実測 IQ による品質セルフテスト

### 1.3 技術スタック

| 分類 | 使用技術 |
|------|---------|
| 言語 | Python 3.11+ |
| 数値計算 | NumPy, SciPy |
| 高速化 | Numba（JIT） |
| SDR | pyrtlsdr（librtlsdr） |
| 音声 | PyAudio（PortAudio） |
| テスト | pytest, GitHub Actions（Ubuntu / Windows） |

### 1.4 動作モード

| モード | サンプルレート | 復調方式 | 用途 |
|--------|--------------|---------|------|
| 標準（Standard） | 1.024 MHz | arctan discriminator（既定）/ PLL | 通常受信 |
| 軽量（Light） | 250 kHz | arctan discriminator | 低 CPU 負荷 |

`--light` の既定はモノラルですが、`stereo on` でステレオに切り替える
と標準モードと同じステレオ復調パイプラインを通ります。

---

## 2. システムアーキテクチャ

### 2.1 全体構成

```
┌─────────────────────────────────────────────────────────┐
│                  FMReceiverController                     │
│  （全サブシステムを統合、facade API を CLI へ提供）        │
├───────────────┬───────────────┬───────────────┬──────────┤
│ SDRReceiver   │ FMDemodulator │ AudioOutput   │ AutoGain │
│ （IQ 取得）    │ （復調）       │ （再生 / 録音） │ Controller│
├───────────────┴───────────────┴───────────────┴──────────┤
│                CommandLineInterface（対話 CLI）            │
└─────────────────────────────────────────────────────────┘
```

### 2.2 スレッド構成

リアルタイム経路（SDR コールバックと処理スレッド）はディスク I/O や
USB 制御転送で決してブロックしないよう、遅い処理はすべて専用ワーカー
スレッドへ分離されています。

| スレッド | 役割 | 生成元 |
|---------|------|--------|
| SDR async read | rtlsdr ライブラリ内部スレッド。IQ ブロックを `data_queue` へ投入 | `SDRReceiver.start()` |
| Processing thread | IQ→復調→音声変換→再生キュー投入 | `FMReceiverController` |
| CLI thread | ユーザ入力の受け付けとコマンド実行 | `CommandLineInterface` |
| Audio callback | PyAudio コールバック。再生バッファへ供給 | PortAudio |
| AudioRecordWorker | 音声 WAV の書き込み（ディスク I/O 隔離） | `AudioOutput` |
| IQRecordWorker | 生 IQ WAV の書き込み（ディスク I/O 隔離） | `SDRReceiver` |
| AutoGainWorker | RTL-SDR の `set_gain()` USB 転送（同期呼び出し隔離） | `AutoGainController` |

### 2.3 リアルタイム安全性の設計

- **SDR data_queue**（最大 80 ブロック ≈ 1.28 秒）: 処理スレッドの
  一時的な遅延を吸収。溢れた場合はサンプルを破棄し WARNING を記録。
- **JIT プリウォーム**: 処理スレッド開始前に Numba / FFT の初回コン
  パイルを空 IQ で済ませ、起動時の ~1.3 秒スパイクによるキュー溢れを
  防止。
- **録音ワーカー**: `record()` / SDR コールバックは bounded queue へ
  投入するのみ。writeframes の 100–1000 ms ストールはワーカー側に閉じ
  込められる。
- **AGC ワーカー**: `set_gain()` の 40–200 ms USB ブロッキングを処理
  スレッドから隔離。要求は coalesce（最新値のみ反映）。

---

## 3. モジュール詳細仕様

### 3.1 コントローラ（controller.py）

`FMReceiverController` が全サブシステムを統合し、CLI へ facade API を
提供します。

主な責務:
- サブシステムの初期化（SDR / 復調器 / 音声出力 / AGC / CLI）
- 処理スレッドの実行（`processing_thread`）と per-block プロファイル
- 選局（`tune`）: 中心周波数変更・キューフラッシュ・復調器リセット・
  録音停止
- JIT プリウォーム（`_prewarm_jit`）
- ステレオ / モノ、録音、IQ 録音、AGC、ゲインの facade メソッド

`_BlockProfiler` が per-block の処理時間・キュー深さを計測し、16 ms
予算を越えるブロックや 60 秒サマリをログ出力します（`--log` 時のみ）。

### 3.2 SDR 受信機（sdr_receiver.py）

`SDRReceiver` が RTL-SDR デバイスからの IQ 取得と、生 IQ 録音を管理し
ます。

- `read_samples_async` によるコールバック駆動取得
- コールバックは complex64 化して `data_queue` へ投入（非ブロッキング）
- 生 IQ 録音は `IQRecordWorker` へ委譲（ワーカー設計は §2.3、メタデータ
  は 3.10 参照）: flush sentinel、4 GiB ローテーション、書き込み例外の
  握り込み、メタデータサイドカー

### 3.3 FM 復調器（demodulator.py）

クラス階層:

```
FMDemodulatorInterface (ABC)
  └─ BaseFMDemodulator（共通フィルタ・復調・リサンプリング）
       ├─ FMDemodulator      （標準: discriminator / PLL）
       └─ FMDemodulatorLight （軽量: discriminator, パイロット LPF のみ order-1）
```

#### FMDemodulator（標準）

`process_iq_samples()`:
1. DC ブロッカ（複素1次 LTI ハイパス、fc=0.1 Hz、状態保持）。旧ブロック
   平均 EMA 減算を置換 — FM ではブロック平均が変調に支配されるため推定が
   揺らぎ、除去された成分が加法誤差となり、その位相方向成分 −Im(c·e^{−jφ})
   が discriminator の非線形性を介して変調相関の積(−22 dB)として composite
   へ現れていた（搬送波オフセット 0 の合成評価における「8–12 kHz セパレー
   ションディップ」の根本原因）。除去帯域を広げるほどこの誤差の入力電力が
   増えるため 0.1 Hz（旧 EMA の実効帯域と同一）を採用し、実キャプチャで
   pilot SNR は EMA と 0.01 dB 以内で一致。病理はノッチ遷移帯
   （おおむね |オフセット| < 0.1–0.3 Hz）で連続的に存在するが、本実機の
   実測残留オフセット ~60 Hz はノッチから十分離れている（合成の周期
   トーンは離散キャリア線がノッチに一致し得るため強く顕在化する）。
   LTI なのでブロック分割不変・0 Hz で厳密ゼロ・整定 ~1.6 s（EMA と同じ）
2. IQ ローパス（Butterworth N=5、SOS、ブロック間状態保持）
3. FM 復調 — 既定は **arctan discriminator**
   `angle(x[n]·conj(x[n-1]))`（前ブロック末尾サンプルを持ち越して
   ブロック境界で連続）。`MAIN_DEMOD_USE_PLL=True` でレガシー PLL 経路
   に切替可能。
4. IQ→composite リサンプル（`StatefulResampler` 3:16、Kaiser β=10）

discriminator を既定とする理由: PLL の閉ループ応答は MPX 帯域で
+3.9 dB のピーキングと 19k–38k 間で -30.7° の位相不整合を持つのに対し、
discriminator は全帯域で平坦・純遅延。合成 IQ でステレオセパレーション
が +4–6 dB 改善する。

#### FMDemodulatorLight（軽量）

`process_iq_samples()` は標準モードと同じ arctan discriminator
（`angle(x[n]·conj(x[n-1]))`）で FM 復調し、`StatefulResampler` 96:125 で
250 kHz→192 kHz に変換、`LIGHT_COMPOSITE_SCALE=0.35` を乗じます。以降の
ステレオ復調は標準モードと共通（mono/side FIR バンクも同一タップ数を共有。パイロット複素 LPF のみ order-1、SNR ノイズ帯 BPF は標準と同じ order-9 — 旧 order-1 流用はパイロット漏れで SNR を ~10 dB に飽和させ、軽量モードが full stereo に到達できなかった）。旧実装の
`angle→unwrap→diff` は unwrap 位相が搬送波オフセット下で無限成長し、
float32 量子化により長時間セッションで劣化するため置換されました
（伝達特性は同一）。

#### ステレオ復調（`_demodulate_stereo`）

`BaseFMDemodulator` の共通処理:
1. モノ抽出（15 kHz 線形位相 FIR、下記フィルタバンク）。mono/side の
   全フィルタが同一タップ数を共有するため両経路の群遅延は構造的に
   一致し、遅延補償は不要（`STEREO_MONO_DELAY_SAMPLES=0`）
2. パイロット抽出 — **ヘテロダイン方式**: composite を 19 kHz で位相
   連続にミックスダウンし、複素ローパス（Butterworth N=9、SOS、状態
   保持）で複素残差を得る。FFT Hilbert を用いない（ブロック端の位相
   グリッチが原理的に発生しない）。
3. パイロット SNR 推定（残差電力 `2·mean(|z|²)` と 2 本のノイズ帯の比）
   と適応ブレンド係数の更新
4. サブキャリア生成 `cos/sin(2θ + φ_offset)`（φ_offset は復調方式ごとの
   DSP 固有値 discriminator 1.0°/PLL 331.1°/Light 0.3° に、実機では
   前段トリム `HARDWARE_SUBCARRIER_PHASE_TRIM_DEG`(+84°)を加算。
   これにより位相トラッカーの実機需要が ±90° 境界から 0° 付近へ移り、
   セッション間の取得枝（L/R 極性）が安定する）
5. L−R 帯の同期復調（DSB-SC、ゲイン 2.0）と 3 バンド LPF（I/Q 並列）。
   **前置バンドパスなし** — 理想的な 38 kHz 同期ミキサと **0–15 kHz の
   対象帯域内**では、raw composite 復調 + 15 kHz LPF は理想 23–53 kHz
   BPF を前置した場合と等価（モノ帯は 23–53 kHz へ、パイロット像は
   19 kHz へ移り LPF 阻止域）。これにより実 Butterworth BPF の帯域端
   特性（12–14 kHz の DSB 側波帯を削り HF セパレーションを崩壊させて
   いた実測要因）が原理ごと除去される。等価性は FIR の 15–18.5 kHz
   遷移帯には及ばず、帯域外 composite 成分（20.5–22 k / 54–56.5 kHz）
   が 16–18.5 kHz へ写像され得るため、最終音声帯域制限（下記手順 11）
   で抑制する。mono/side/3 バンドの全 7 フィルタ
   は同一タップ数（192 kHz で 321、群遅延 160 サンプル = 0.83 ms）の
   線形位相 FIR（Kaiser β=9、遷移 3.5 kHz、阻止域 ~−100 dB、19 kHz で
   実測 −102 dB）で、overlap-save（タップスペクトル事前計算）により
   任意ブロック長でワンショット畳み込みと厳密一致
6. I/Q 位相補正 — **ゲート付き4象限トラッカー**: 主軸推定
   `½·atan2(2·Cov, Var_I−Var_Q)` は ±90° 周期の曖昧性を持つため、
   (a) 共分散異方性 ≥ `STEREO_PHASE_ANISO_GATE`(0.2)、
   (b) side 電力 ≥ mono 電力 × `STEREO_PHASE_SIDE_GATE_DB`(-18 dB)、
   (c) side 電力 ≥ ノイズ推定 × `STEREO_PHASE_SIDE_OVER_NOISE_DB`
   (26 dB — FM ノイズは f² スペクトルにより side 帯で mono 帯より
   強く、かつ帯域非対称性で異方的な**疑似軸**を成すため、pilot ノイズ
   帯からの予測値で本物の side 成分と弁別)の全条件を満たすブロック
   のみで、π 周期族のうち現追跡値に最も近い候補への差分で EMA を更新
   （連続性が 180° 分岐を解決、クランプ不要で ±180° 追従）。更新の
   重みは異方性とノイズマージンの積（限界的ブロックはほぼ無重量）。
   ±90° 跨ぎ（= L/R 極性反転）は信頼度 EMA ≥
   `STEREO_PHASE_BRANCH_CONF`(0.7)の持続的な高信頼追跡時のみ許可。
   ゲート閉時は取得済みの角度を `STEREO_PHASE_LEAK_DEG_PER_SEC`
   (0.5°/s)で 0（トリム事前値）へ減衰。コールドスタート取得は連続
   `STEREO_PHASE_ACQUIRE_BLOCKS`(6)ブロックの倍角円平均
   `½·arg(Σe^{j2β})` で初期化（個々の生推定の ±90° ラップに不変）。
   取得時のみ「真の回転が ±90° 内」という FM 規格の pilot 位相規約を
   仮定
7. 3 バンド整形 + ブレンド + HF ブレンドの上限ゲイン（既定は中立 1.0）
8. ステレオマトリクス `L=Mono+Side, R=Mono−Side`
9. パイロットノッチ ×2（19 kHz、Q=30）
10. composite→audio リサンプル（`StatefulResampler` 1:4、L/R 独立）
11. 最終音声帯域制限 — L/R に**同一タップ**の線形位相 FIR
    （48 kHz で 183 taps、通過域 15 kHz / 阻止域 16.5 kHz、遅延 1.9 ms。
    手順 5 の遷移帯写像成分を抑制。同一フィルタのため
    セパレーションへは影響しない。モノ動作時も両インスタンスを
    同一入力で進め、L/R のサンプル数・出力グリッド・LPF state の
    不整合を防ぐ）
12. ディエンファシス（τ=50 μs）
13. **Side NR**（既定 ON、下記）

#### Side チャネル雑音抑制

`SideNoiseReducer`（3.5 参照）が音声出力段で mid=(L+R)/2 を無加工の
まま、side=(L−R)/2 のみを DD-Wiener STFT で雑音抑制します。モノ成分の
音色は完全保存し、ステレオ由来の HF ヒスのみを狙って除去します。

**mono↔stereo 切替連続性**（issue #29 対応）: mid/side NR テールは
stereo・mono 両経路で共有され（`_apply_side_nr`。モノでは side=(L−R)/2≈0
が流れる）、モノ動作中も NR 連鎖と mid アライナが前進します。これにより
(a) stereo→mono 切替では保持中の旧 side が NR オーバーラップを通じて
滑らかにフラッシュされ（旧実装は破棄後に stereo 復帰時 stale 再生:
実測 side RMS 0.31/0.19/床）、(b) 出力レイテンシがモード非依存になり
切替毎の ~16 ms タイムジャンプが消滅。さらに stereo 復帰時は
`_reset_stereo_side_state` が stereo 専用状態（L−R FIR バンク、パイロット
ヘテロダイン/PLL、ノイズ帯 BPF、blend/トラッカーの取得状態）を初期同調と
同じ再取得semanticsでクリアし、復帰 block 0 から stale side なし（最悪
条件 blend 強制 1 で実測 4e-4、適応 blend では ~5e-5）。モノ動作中の NR は
`adapt=False`（時間機構のみ前進する厳密パススルー）で駆動され、
スペクトル適応状態（noise floor / power_smooth / DD state）は凍結される —
人工的な side≈0 に最小統計を適応させると床が数秒で崩壊し
（実測: モノ 4 秒で −144 dB）、復帰後 30–50 秒 NR が実質無効になるため。
凍結によりモノ区間の長さに関係なく学習済みモデルが保存され、復帰直後から
NR が有効。残渣は切替を跨ぐ ~2 フレームの床ディップのみ（power smoothing
EMA で有界、上方リークで 1 秒以内に回復）。

### 3.4 音声出力（audio_output.py）

`AudioOutput` が PyAudio による再生と、音声 WAV 録音を管理します。

- コールバック駆動再生（`_buffer_deque` でゼロコピー供給）
- 録音は `AudioRecordWorker` へ委譲（ワーカー設計は §2.3 参照）
- `record()` は bounded queue への投入のみで即時復帰
- start / stop / rotation / sidecar は IQ 録音と共通の設計（メタデータは
  3.10 参照）

### 3.5 フィルタ（filters.py）

| クラス | 種別 | 備考 |
|--------|------|------|
| `LowpassFilter` | Butterworth LPF（SOS） | ストリーミング、状態保持、`reset()` |
| `FIRFilter` | 線形位相 FIR（overlap-save） | mono/side フィルタバンクと、マトリクス後の L/R 共通・最終音声 LPF の両方で使用。タップスペクトル事前計算、ストリーミング状態保持、`reset()` 対応、任意ブロック長でワンショットと厳密一致 |
| `BandpassFilter` | Butterworth BPF（SOS） | 同上 |
| `NotchFilter` | IIR ノッチ | 同上 |
| `DeemphasisIIRFilter` | ディエンファシス IIR | Numba 最適化 |
| `StatefulResampler` | polyphase リサンプラ | グリッド整合、任意ブロック長で厳密 prefix |
| `SideNoiseReducer` | DD-Wiener STFT 雑音抑制 | mid/side の side 専用 |
| `StreamAligner` | ストリーム整合バッファ | side NR の mid 遅延補償 |

#### StatefulResampler

`scipy.signal.resample_poly` はステートレスでブロック端をゼロパッドし、
ブロック境界に過渡ノイズを生みます。本クラスは:
- 前ブロックの履歴を可変長で前置し、拡張ブロックの開始を常に polyphase
  グリッド周期（`down/gcd(up,down)`）の倍数に整合（任意ブロック長・比で
  厳密に一括処理の prefix と一致）
- FIR サポートが受信済み入力で完結した出力のみ放出（末尾の未確定出力を
  持ち越し）。定数レイテンシは half-filter 長。
- `emit_align` で放出境界を下流間引き係数の倍数に丸める

#### SideNoiseReducer

Mid/Side の side チャネルに対する STFT スペクトル雑音抑制:
- STFT: frame 1024 / hop 256（75% オーバーラップ、Hann 解析+合成窓、
  レイテンシ ~16 ms）
- ノイズ床: 平滑パワーの最小統計 + バイアス補正。トラッカー入力は周波数
  方向の局所中央値 +8 dB でクランプ(トーン保護 — 定常トーンが自ビンの
  床に吸収されて `alpha_floor` まで削られるのを防止。広帯域ノイズの推定
  挙動は不変)
- Ephraim-Malah Decision-Directed（α_dd=0.98）による a priori SNR 推定
  → Wiener ゲイン（下限 `alpha_floor`、既定 0.30 = -10 dB）
- 周波数方向 3-bin ゲイン平滑（musical noise 抑制）
- 動作帯域 1.5–15 kHz（低域のステレオ定位を保存）

### 3.6 PLL（pll.py）

`PLL` クラス（Numba 最適化）。パイロット位相追跡（`return_phase=True`）
に使用。標準の FM 復調は既定で discriminator のため、メイン PLL は
`MAIN_DEMOD_USE_PLL=True` の場合のみ使用されます。

### 3.7 自動ゲイン制御（auto_gain.py）

`AutoGainController` が RTL-SDR の手動ゲインステップを用いたソフトウェア
AGC を実装します。

- IQ ピーク振幅を監視し、クリップ / 弱信号が連続すると 1 ステップ調整
- `set_gain()` の USB 転送は `AutoGainWorker` スレッドへ非同期投入
  （処理スレッドをブロックしない、要求は coalesce）
- `disable()` は現ゲインに pin（AGC 保留要求で後から変わらない）
- 起動後 `AGC_WARMUP_SEC`（既定 2.0 秒）は AGC を抑止（JIT スパイク中の
  誤発火を防止）

### 3.8 CLI（cli.py）

`CommandLineInterface`（スレッド）がユーザコマンドを facade API 経由で
実行します。ディスパッチは 1) 完全一致 → 2) プレフィックス一致（agc /
gain）→ 3) 数値入力（局番号 / 周波数）の順。録音ファイル名は
`build_recording_path()` が `recordings/` 下に自動生成します。

### 3.9 品質セルフテスト（quality_selftest.py）

合成 IQ または実測 IQ WAV に対する客観指標の測定モジュール。

- 合成経路: 既知ステレオ音声 → MPX 合成 → FM 変調 IQ → 復調 → 指標
  - THD+N、SNR（参照ベストフィット比）、ステレオセパレーション
  - 指標は 1 秒窓・0.5 秒ホップの**窓中央値**（窓端の過渡に頑健）
  - チャネル障害モデル: `--clock-ppm`、`--carrier-offset-hz`、
    `--multipath-*`、プリエンファシス（既定 ON）
- 実測 IQ 経路（`--iq-wav`）: パイロット SNR 分位点、mid/side の HF
  ノイズ床、ステレオ聴取ペナルティ、CSV 出力
- セパレーション周波数特性（`--sep-sweep`）: **高忠実リファレンス変調器**
  （全 MPX 成分を IQ レートで解析的に直接合成 — リサンプリング皆無、
  pre-emphasis はトーン毎に閉形式適用、位相は台形積分、変調率一定
  正規化）で周波数別のセパレーション/THD を測定。`--hifi-tx` で通常
  評価にも同変調器を使用可能。旧来の
  「Sep ~30dB 床」の主因は blend の stability 項(中立化済み)で
  あり、hifi TX は送信側リサンプリングの影響を排除して blend 中立
  化後の高ダイナミックレンジ測定(1kHz で 43-57dB)での送受切り分け
  を可能にする
- 周波数応答経路（`--sweep-response`）: 単一トーンを対数間隔で掃引し
  （無雑音）、mono（L+R）/ side（L−R）経路の振幅応答を dB テーブルで
  出力。ディエンファシスのロールオフ、15 kHz LPF、19 kHz ノッチの効き、
  side NR の定常成分減衰などを検証できる。`--preemphasis` /
  `--no-preemphasis` と `--side-nr*` を尊重。

### 3.10 録音メタデータ（recording_meta.py）

録音セッションごとに `<base>.json` サイドカーを生成する共通モジュール。
録音開始時に捕捉パラメータ（周波数・ゲイン・レート）を、停止時に
タイムスタンプ・パート一覧（4 GiB ローテーションの分割ファイル群）・
ドロップ数を確定書き込みします。JSON はファイルを開く前に完全シリアラ
イズし（部分書き込み防止）、シリアライズ失敗・書き込み失敗のいずれも
録音を止めません（NumPy スカラーは `.item()` 変換）。

### 3.11 エントリポイント（__main__.py）

`python -m fm_radio` / `fm_receiver.py` の起点。コマンドライン引数
（`--light` / `--log`（別名 `--verbose` / `-v`）/ `--debug` /
`--log-file`）を解釈し、ログ設定を
行い（有効時）、`FMReceiverController` を生成して `start()` を呼びます。
ログ有効時は GC モニタ（Gen2 コレクションのポーズ時間を記録）も登録し、
音声ドロップ調査に用います。

### 3.12 ログ設定（logging_config.py）

`setup_logging()` がフォーマットとハンドラ（コンソール、任意でファイル）
を構成します。ログはオプトインで、無効時は `logging.disable(CRITICAL)`
により全ロガーが抑止されます（既定はログ無効）。

---

## 4. データフロー

### 4.1 IQ サンプルフロー

```
RTL-SDR ─(async callback)→ data_queue ─(processing thread)→
  process_iq_samples ─→ demodulate ─→ enqueue_audio ─→
  AudioOutput.audio_buffer_queue ─(PyAudio callback)→ 再生
```

処理スレッドは録音有効時、復調済みステレオを `AudioOutput.record()` へも
渡します（bounded queue 経由でワーカーが書き込み）。

### 4.2 音声出力フロー

```
(L, R) ─→ audio_buffer_queue ─(callback)→ _buffer_deque ─→
  インターリーブ float32 ─→ PortAudio 出力
```

録音有効時は並行して:

```
(L, R) ─→ _record_q ─(AudioRecordWorker)→
  int16 変換 ─→ writeframes（4 GiB でローテーション）
```

---

## 5. 信号処理パイプライン

### 5.1 標準モード処理チェーン

```
IQ (1.024 MHz)
  → DC ブロッカ (複素1次 LTI ハイパス, fc=0.1 Hz)
  → IQ LPF (Butterworth N=5, SOS, 状態保持, fc=200 kHz)
  → FM 復調 (arctan discriminator 既定 / PLL 選択可)
  → StatefulResampler 3:16 (Kaiser β=10) → composite (192 kHz)
  → ステレオ復調:
      ├ モノ: 線形位相 FIR (321 taps, 15 kHz)
      ├ パイロット: 19 kHz ヘテロダイン + 複素 LPF (N=9) → 位相 θ
      ├ サブキャリア cos/sin(2θ + φ_offset)
      ├ composite × サブキャリア × 2.0 (前置 BPF なし)
      ├ 3 バンド線形位相 FIR (I/Q, モノと同一タップ数=同一群遅延)
      ├ IQ 位相補正 (異方性ゲート付き4象限トラッカー)
      ├ 3 バンド整形 + 適応ブレンド + HF ブレンド上限
      └ マトリクス L=M+S, R=M−S
  → パイロットノッチ ×2 (19 kHz, Q=30)
  → StatefulResampler 1:4 → audio (48 kHz)
  → 最終音声帯域制限 (L/R 共通 FIR, 15 k/16.5 k, 183 taps)
  → ディエンファシス (τ=50 μs)
  → Side NR (mid/side DD-Wiener STFT, 既定 ON)
  → (L, R) 48 kHz
```

### 5.2 軽量モード処理チェーン

```
IQ (250 kHz)
  → DC ブロッカ (標準と同一)
  → FM 復調 (arctan discriminator, 標準と同一)
  → StatefulResampler 96:125 → × 0.35 → composite (192 kHz)
  → ステレオ復調 (標準と共通の FIR バンク; パイロット LPF のみ order-1)
```

---

## 6. 設定パラメータ

すべて `fm_radio/constants.py` に定義。主要なもののみ抜粋します。

### 6.1 SDR 設定

| 定数 | 値 | 説明 |
|------|-----|------|
| `SDR_SAMPLE_RATE` | 1.024e6 | 標準モードサンプルレート |
| `SDR_SAMPLE_RATE_LIGHT` | 0.25e6 | 軽量モードサンプルレート |
| `SDR_BLOCK_SIZE` | 16384 | 1 読み出しあたりのサンプル数 |
| `SDR_QUEUE_MAXSIZE` | 80 | data_queue 最大ブロック数（≈1.28 秒） |

### 6.2 FM 復調 / PLL 設定

| 定数 | 値 | 説明 |
|------|-----|------|
| `MAIN_DEMOD_USE_PLL` | False | True で PLL 復調、False で discriminator |
| `MAIN_PLL_KP` / `MAIN_PLL_KI` | 0.12926 / 0.0208844 | メイン PLL ゲイン（PLL 選択時のみ） |
| `PILOT_PLL_KP` / `PILOT_PLL_KI` | 0.032 / 0.00008 | パイロット PLL ゲイン |

### 6.3 フィルタ設定

| 定数 | 値 | 説明 |
|------|-----|------|
| `IQ_LOWPASS_ORDER` / `IQ_LOWPASS_CUTOFF` | 5 / 200 kHz | IQ ローパス |
| `MONO_LOWPASS_CUTOFF` | 15 kHz | モノ / L−R ベース LPF カットオフ |
| `PILOT_BANDPASS_ORDER` | 9 | パイロット複素 LPF 次数（軽量は 1 — サブキャリア位相動作点の校正対象） |
| `PILOT_NOISE_BAND_ORDER` | 9 | パイロット SNR ノイズ帯 BPF 次数（**両変種共通**。旧実装で軽量が order-1 を流用した際、スカートがパイロット自身をノイズ参照へ漏らし SNR が 9.975 dB / blend 0.313 に飽和していた） |
| `STEREO_FIR_NTAPS` / `_TRANSITION_HZ` | 321 / 3.5 kHz | mono/side 線形位相 FIR バンク（タップ数は composite レート比例、全フィルタ共有） |
| `AUDIO_FINAL_LP_NTAPS` / `_CUTOFF_HZ` / `_STOP_HZ` | 183 / 15 k / 16.5 k | 最終音声帯域制限（L/R 共通 FIR、raw composite 復調の遷移帯写像を抑制） |
| `DEEMPHASIS_TAU` | 50e-6 | ディエンファシス時定数 |
| `PILOT_NOTCH_FREQ` / `_Q` | 19 kHz / 30 | パイロットノッチ |

### 6.4 ステレオブレンド / 位相補正設定

| 定数 | 値 | 説明 |
|------|-----|------|
| `STEREO_BLEND_PILOT_SNR_DB_LO` / `_HI` | 7.0 / 16.5 | ブレンド係数の SNR 範囲 |
| `STEREO_HF_BLEND_PILOT_SNR_DB_LO` / `_HI` | 15.0 / 35.0 | HF ブレンド上限の SNR ランプ |
| `LR_HIGH_MAX_GAIN` / `LR_SUPER_HIGH_MAX_GAIN` | 1.00 / 1.00 | HF 減衰上限（既定は中立） |
| `STEREO_PHASE_ANISO_GATE` | 0.2 | 位相トラッカー更新に要する共分散異方性（音楽 p5=0.55 / ノイズ p99=0.05 の間） |
| `STEREO_SUBCARRIER_PHASE_OFFSET_DEG` | 1.0 | サブキャリア位相オフセット（discriminator、DSP固有値。FIR バンク + 前置 BPF 撤去で旧 316° の大半が不要となりほぼ 0 に） |
| `HARDWARE_SUBCARRIER_PHASE_TRIM_DEG` | 84.0 | 実機前段（チューナ IF）の位相トリム。全変種の DSP 値に加算（合成経路は非適用）。実測: アンテナ2局+光伝送の全実録音が同一の ~±85-90° 需要を示し、マルチパスではなく前段特性と同定 |
| `STEREO_SUBCARRIER_PHASE_OFFSET_DEG_PLL` | 331.1 | 同（PLL 選択時） |
| `STEREO_SUBCARRIER_PHASE_OFFSET_DEG_LIGHT` | 0.3 | 同（軽量モード） |
| `STEREO_MONO_DELAY_SAMPLES` | 0 | モノ遅延補償（FIR バンクの群遅延一致により不要） |
| `STEREO_BLEND_FAST_CLOSE_FACTOR` | 0.5 | ドロップアウト検出時の blend 減衰（16 ms 基準ブロックあたり） |
| `STEREO_BLEND_FAST_CLOSE_SETTLE_REF` | 12.0 | fast-close を許可するまでの整定時間（基準ブロック数 ≈ 190 ms） |
| `STEREO_BLEND_DROPOUT_SNR_DEBOUNCE_REF` | 12.0 | 瞬時 SNR が LO 未満のまま継続したら持続的劣化と判定する時間（基準ブロック数 ≈ 190 ms。実プログラムの連続 LO 割れは最長 131 ms＝軽量ブロック量子化後） |
| `STEREO_BLEND_DROPOUT_POWER_DROP_DB` | 15.0 | パイロット電力が自 EMA から崩落したと判定する量（`pow_drop_db = 10log10(EMA / 現在値)`。実プログラムでの最大電力降下は +1.13 dB） |

**ブレンド EMA の時間正規化とドロップアウト fast-close**: blend / パイロット
SNR 系の EMA 係数はすべて 16 ms 基準ブロックに正規化されます
（`alpha_eff = 1 − (1 − alpha)^(block / 16 ms)`）。時間刻みは IQ ブロック長
（`iq_samples.size / iq_sample_rate`）から取り、`process_iq_samples()` 1 回に対する
`demodulate()` 1 回だけが消費します（composite 直接呼び出しは composite 長へ
フォールバック）。これにより、実運用ブロックが 16384 サンプル @ 250 kHz
（≈65.5 ms）の軽量モードでも、標準モード（16384 @ 1.024 MHz = 厳密に 16 ms）と
同一の時定数になります。加えて、パイロット劣化時は 3 系統の検出器
——(a) パイロット電力が自 EMA 比 `STEREO_BLEND_DROPOUT_POWER_DROP_DB` 以上の崩落、
(b) 瞬時と EMA の両パイロット SNR が `STEREO_BLEND_PILOT_SNR_DB_LO` 未満、
(c) 瞬時 SNR が LO 未満のまま `STEREO_BLEND_DROPOUT_SNR_DEBOUNCE_REF` 基準ブロック
継続——のいずれかが成立し、かつ整定ガードを越えている間、blend を基準ブロック
あたり `STEREO_BLEND_FAST_CLOSE_FACTOR` 倍で閉じます。(c) は「パイロットは
健在のまま雑音床だけが上がる」劣化のための経路で、(a) は電力が変わらないため
発火せず、(b) は遅い SNR EMA が LO を割るまで約 0.65 s かかります。デバウンス
なしの瞬時 SNR 単独トリガは実音楽のノイズ帯スピルで誤作動するため不採用
（実測: 4 つの参照録音 60 s ずつでの最長連続 LO 割れは 16 ms ブロックで 48 ms
(CATV) / 80 ms (光 82.5)、軽量の 65.5 ms ブロックでは量子化されて 131 ms。
デバウンス 192 ms はこれを ~1.5 倍で上回り、どの録音でも発火しない）。
実測: 軽量モードの実運用ブロックで、パイロットレス同調・ドロップアウトとも
0.197 s で閉鎖、雑音床ステップは blend&lt;0.5 が 0.197 s / blend&lt;0.05 が 0.262 s
（対策前は 0.524 / 0.655 s）、復帰は blend&gt;0.5 が 0.197 s / blend&gt;0.9 が 0.524 s。標準モードの通常ブロックは指数が厳密に 1.0 と
なる恒等ショートカットにより、従来と bit 同一です。

### 6.5 Side NR 設定

| 定数 | 値 | 説明 |
|------|-----|------|
| `SIDE_NR_ENABLE` | True | side チャネル雑音抑制の有効化 |
| `SIDE_NR_FRAME` / `SIDE_NR_HOP` | 1024 / 256 | STFT フレーム / ホップ |
| `SIDE_NR_ALPHA_FLOOR` | 0.30 | Wiener ゲイン下限（≈-10 dB） |
| `SIDE_NR_LO_HZ` / `SIDE_NR_HI_HZ` | 1500 / 15000 | 動作帯域 |

### 6.6 音声出力・録音設定

| 定数 | 値 | 説明 |
|------|-----|------|
| `AUDIO_OUTPUT_RATE` | 48000 | 音声出力サンプルレート |
| `RECORDINGS_DIR` | "recordings" | 録音・サイドカーの保存先 |
| `RECORD_QUEUE_MAXSIZE` / `IQ_RECORD_QUEUE_MAXSIZE` | 200 / 200 | 録音ワーカーキュー |
| `AUDIO_RECORD_ROTATE_THRESHOLD_BYTES` | 4e9 | WAV ローテーション閾値 |
| `IQ_RECORD_ROTATE_THRESHOLD_BYTES` | 4e9 | 同（IQ 録音） |

### 6.7 AGC 設定

| 定数 | 値 | 説明 |
|------|-----|------|
| `AGC_DEFAULT_GAIN_INDEX` | 19 | 起動時ゲインインデックス（36.4 dB） |
| `AGC_CLIP_THRESHOLD` / `AGC_WEAK_THRESHOLD` | 0.95 / 0.3 | クリップ / 弱信号閾値 |
| `AGC_CLIP_COUNT` / `AGC_WEAK_COUNT` | 3 / 15 | 調整発火の連続ブロック数 |
| `AGC_WARMUP_SEC` | 2.0 | 起動後の AGC 抑止時間 |

---

## 7. インターフェース仕様

`fm_radio/interfaces.py` に抽象基底クラスを定義します。

| インターフェース | 実装 | 主なメソッド |
|-----------------|------|-------------|
| `SDRReceiverInterface` | `SDRReceiver` | `start`, `stop`, `set_center_frequency`, `start_iq_recording` |
| `FMDemodulatorInterface` | `FMDemodulator`, `FMDemodulatorLight` | `process_iq_samples`, `demodulate`, `reset` |
| `AudioOutputInterface` | `AudioOutput` | `enqueue_audio`, `start_recording(…, metadata)`, `record`, `cleanup` |

例外は `fm_radio/exceptions.py` に定義（`SDRDeviceError`,
`DemodulationError`, `AudioOutputError`, `RecordingError`）。

---

## 8. 使用方法

### 8.1 基本的な起動

```bash
# 標準モード
python fm_receiver.py

# 軽量モード
python fm_receiver.py --light

# ログ有効化 / デバッグ / ファイル出力
python fm_receiver.py --log        # 別名: --verbose / -v
python fm_receiver.py --debug
python fm_receiver.py --log-file fm_receiver.log
```

### 8.2 対話コマンド

| コマンド | 動作 |
|---------|------|
| `list` | 局リスト表示 |
| `<番号>` / `<周波数MHz>` | 選局 |
| `stereo on` / `stereo` | ステレオ復調 |
| `stereo off` / `mono` | モノラル復調 |
| `record start` / `record stop` | 音声録音（`recordings/` に自動命名） |
| `iqrec start` / `iqrec stop` | 生 IQ 録音 |
| `agc on` / `agc off` | AGC 有効 / 無効 |
| `gain <値>` | 手動ゲイン設定（AGC 無効時） |
| `q` | 終了 |

### 8.3 品質セルフテスト

```bash
# 合成 IQ（THD+N / SNR / セパレーション）
python -m fm_radio.quality_selftest --duration 5 --cnr-db 35

# チャネル障害を付与
python -m fm_radio.quality_selftest --duration 3 --cnr-db 35 --clock-ppm 200

# 実測 IQ WAV の診断
python -m fm_radio.quality_selftest --iq-wav recordings/<file>.wav \
    --duration 30 --warmup-s 2.5

# 音声経路の周波数応答（mono / side、dB テーブル）
python -m fm_radio.quality_selftest --sweep-response --duration 2
python -m fm_radio.quality_selftest --sweep-response --no-side-nr  # side NR オフ比較
```

---

## 9. 録音ファイル

自動命名の録音（音声・生 IQ とも）は `recordings/` ディレクトリに保存
されます。各セッションには WAV と並んで `.json` メタデータサイドカーが
生成され、捕捉パラメータ（中心周波数・ゲイン・サンプルレート）、開始 /
停止タイムスタンプ、ドロップ数、そして 4 GiB ローテーションで分割された
`.partNNN.wav` の一覧が記録されます。

WAV フォーマットはデータチャンクサイズを 32-bit で保持するため単一
ファイルは 4 GiB が上限です。閾値手前で自動的に次ファイル（`foo.wav`
→ `foo.part001.wav` → …）へ切り替わります。1.024 Msps / 16-bit / 2ch
の生 IQ で約 16 分ごと、48 kHz 音声で約 6.2 時間ごとにローテーション
します。

サイドカー例:

```json
{
  "type": "iq",
  "file": "20260510_213606_80.0MHz_IQ.wav",
  "sample_rate_hz": 1024000,
  "center_freq_hz": 80000000.0,
  "gain_db": 20.7,
  "started_at": "2026-05-10T21:36:06+09:00",
  "stopped_at": "2026-05-10T21:54:51+09:00",
  "parts": [
    "20260510_213606_80.0MHz_IQ.wav",
    "20260510_213606_80.0MHz_IQ.part001.wav"
  ],
  "dropped_blocks": 0
}
```

---

## 10. テストと CI

`tests/` に pytest スイートを配置します。`tests/conftest.py` が
`pyaudio` / `rtlsdr` のフェイクを `fm_radio` インポート前に注入するため、
サウンドデバイス・PortAudio・librtlsdr なしで実行できます。

| テストファイル | 対象 |
|---------------|------|
| `test_filters.py` | フィルタリセット、リサンプラのグリッド整合と厳密 prefix、Side NR |
| `test_demodulator.py` | IQ LPF / pilot / discriminator / ブロックサイズ不変性 |
| `test_auto_gain.py` | 非同期 AGC、ゲイン pin、ウォームアップ |
| `test_audio_output.py` | 非同期録音、フラッシュ、並行 start、ローテーション |
| `test_sdr_receiver.py` | IQ 録音の同等保証、コールバック非ブロッキング |
| `test_quality_selftest.py` | mmap ローダ、障害生成、メトリック頑健性 |
| `test_recording_meta.py` | サイドカー生成・確定、パート一覧 |
| `test_e2e_quality.py` | 合成 MPX→IQ→復調の品質ゲート（slow） |

実行:

```bash
pip install -r requirements-dev.txt
pytest              # 全テスト
pytest -m "not slow"  # E2E 品質ゲートを除く
```

GitHub Actions（`.github/workflows/ci.yml`）が `compileall` + `pytest`
を Ubuntu / Windows・Python 3.11 で、`main` への push と全 PR に対して
実行します。
