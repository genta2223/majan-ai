# 🀄 Mahjong AI Agent (Tactical Push-Fold Engine)

深層強化学習ベースの麻雀AI（Mortal / libriichi等）をコアエンジンとして活用し、**「3秒以内（推論200ms以内 + 人間らしい思考ディレイ1.0〜3.0秒）」** で最適な打牌・副露判断を返し、**順位・点差・局況（オーラス等）に応じた動的な押し引き（攻撃・守備バランス）** を調整可能なモジュラー型麻雀AI推論基盤です。

**フェーズ4では「半自動アシスト操作モード (`semi_auto`)」および「友人牌譜プロファイリング＆特性適応エクスプロイト (`core/opponent_profiler.py`)」** を追加統合しました。

---

## 1. 特徴

- **Windowsアプリ版（デスクトップ版）雀魂の完全サポート**:
  - **CDPアタッチ方式 (`--live-cdp`)**: `--remote-debugging-port=9222` 付きで起動されたデスクトップアプリの Chromium DevTools Protocol にアタッチし、WebSocketフレームを直接抽出。
  - **ローカルプロキシ連携 (`--live-mitm`)**: `tools/mitm_addon.py` による mitmproxy 連携ブリッジを同梱。
- **3つの出力モード (`output_mode`)**:
  1. **`dashboard` (見るだけアシスタント)**: 推奨打牌トップ3（Modified Q値・Raw Q値・危険度・シャンテン数）、局況、リーチ警戒をTUI HUDにリアルタイム表示。マウス操作なし。
  2. **`semi_auto` 【新機能: 半自動アシスト】**: AIが推奨牌までベジェ曲線でカーソルを移動させ、**牌の上にホバー（ポップアップ状態）させたまま待機**。ユーザーが目視確認して手動クリックするかSpaceキーで打牌。誤爆を100%防止。
  3. **`auto_click` (完全自動操作)**: PC版雀魂仕様に完全準拠した4ステップマウス制御（ベジェ移動 $\rightarrow$ ホバー待機 $\rightarrow$ シングルクリック $\rightarrow$ 安全圏退避）。
- **友人対戦プロファイリング & 特性適応エクスプロイト (`opponent_profiler.py`) 【新機能】**:
  - 複数牌譜データから対戦相手の癖（攻撃性 `aggressiveness`、副露率 `open_rate`、放銃率 `deal_in_rate`、アーキタイプ）を自動解析。
  - **動的エクスプロイト補正**:
    - **超攻撃型（Berserker Pusher）相手**: 愚形・安手リーチが多いため過度なベタオリを抑制し、カウンター（押し返し）を強化。
    - **超守備型（Iron Wall）相手**: 本手・良形リーチのため、警戒度とベタオリ倍率を通常より大幅に引き上げて放銃を徹底回避。
- **牌譜URLからの直接ダウンロード＆HUD再現 (`tools/paifu_to_mjai.py`)**:
  - 雀魂の対局URL（`https://game.mahjongsoul.com/?paifu=...`）から生データを取得し、`mjai` 形式へ自動変換してHUDシミュレーションを実行。
- **Windowsクライアント描画領域の自動検出 & キャリブレーション (`tools/calibrate_ui.py`)**:
  - Win32 API (`GetClientRect`, `ClientToScreen`) により、タイトルバーや枠線を除いた純粋なゲーム描画領域を自動検出。16:9アスペクトフィット比率計算。

---

## 2. ディレクトリ構成

```text
mahjong_ai_agent/
├── config/
│   ├── strategy_config.yaml   # 押し引きパラメータ、思考時間、マウス制御、キャリブレーション座標設定
│   └── opponent_profiles.json # 【新規】解析された対戦相手のプロファイルデータベース
├── core/
│   ├── __init__.py
│   ├── state_manager.py       # mjai形式イベント解析、盤面状態・現物安全牌・シャンテン数管理
│   ├── engine.py              # Mortal (PyTorchモデル) / 高速ヒューリスティックによるQ値算出
│   ├── strategy_modifier.py   # 順位・点差・局況・対戦相手エクスプロイトに応じたQ値動的補正
│   └── opponent_profiler.py   # 【新規】友人牌譜プロファイリング & 特性抽出モジュール
├── io_adapter/
│   ├── __init__.py
│   ├── input_simulator.py     # テスト用の牌譜/局面ストリーム入力アダプタ
│   ├── action_controller.py   # 思考ラグ演出、PC雀魂マウス操作計画(semi_auto対応)、HUD
│   ├── majsoul_bridge.py      # 雀魂(Liqi)形式とmjai形式の相互変換ブリッジ
│   ├── liqi_proto_decoder.py  # LiqiバイナリProtobuf動的デコーダ
│   ├── live_sniffer.py        # CDP / mitmproxy リアルタイム通信キャプチャ
│   └── stream_client.py       # リアルタイム通信(stdio / WebSocket) & 即時スルーガード
├── tools/
│   ├── profile_opponent.py    # 【新規】複数牌譜から対戦相手プロファイルを自動抽出・保存するCLI
│   ├── paifu_to_mjai.py       # 牌譜URLダウンロード＆mjai一括変換ツール
│   ├── calibrate_ui.py        # Windowsクライアント領域自動検出 & 座標キャリブレーション
│   └── mitm_addon.py          # mitmproxy WebSocketキャプチャアドオン
├── tests/
│   ├── __init__.py
│   └── test_scenarios.py      # 全機能（全12項目）の自動単体テスト
├── main.py                    # エントリーポイント（semi_auto / live / paifu / replay）
├── requirements.txt
└── README.md
```

---

## 3. 実行コマンド集

### ① 半自動アシストモードで対局（最も安全・推奨）
```bash
# Windowsアプリ版雀魂と連動し、推奨牌へカーソル移動＆ホバー停止して人間のクリックを待機
python main.py --live-cdp --port 9222 --output-mode semi_auto --enable-mouse
```

### ② 友人牌譜プロファイリングの実行
```bash
# output/ フォルダ内の牌譜を一括解析し、対戦相手の癖をデータベース化
python tools/profile_opponent.py --dir output
```

### ③ 牌譜URLからダウンロード＆HUD再生
```bash
python main.py --paifu-url "https://game.mahjongsoul.com/?paifu=261016-64fee530-a705-4f80-b5ab-23d9eb7c6b36_a804191042" --output-mode dashboard
```

### ④ ウィンドウ座標キャリブレーション＆ホバー確認
```bash
python tools/calibrate_ui.py --auto-detect --test-hover
```

---

## 4. テストスイートの実行 (全12項目を自動検証)

```bash
pytest -v tests/test_scenarios.py
```

- **テスト1〜4:** 推論速度（<200ms）、オーラス1位ベタオリ、オーラス4位勝負、思考遅延制御
- **テスト5:** 雀魂表記（`0m`, `1z`〜`7z`）と `mjai` 表記の相互変換
- **テスト6:** 1巡ストリーム即時スルー（他家手番はウェイト0秒で即スルー）
- **テスト7:** PC雀魂4ステップマウスシーケンス
- **テスト8:** LiqiバイナリProtobufフレームの動的パース & `mjai` 変換
- **テスト9:** Windowsクライアント領域の自動スケーリング & 16:9アスペクトフィット計算
- **テスト10:** 牌譜URLパース & mjai一括変換パイプライン検証
- **テスト11 【新機能】:** 半自動モードでの「ベジェ移動 $\rightarrow$ ホバー停止待ち」の挙動検証
- **テスト12 【新機能】:** 複数牌譜からのプロファイル算出（攻撃型/守備型判定）＆エクスプロイト補正検証

---

## 5. GUIランチャー & 常時最前面HUD (`gui.py`)

Python標準のTkinterによる、コンパクトなダークテーマHUDです。
- **デバッグ用Chromeのワンクリック起動**（`--remote-debugging-port=9222`）
- **動作モード切替**: `テキストのみ (安全)` / `半自動マウス誘導`
- **リアルタイムHUD**: 打牌推奨（牌名・期待値・向聴数）と局況（自席・本場・供託・残り巡目）の常時最前面表示
- **牌譜解析 & 相手の癖を読んだエクスプロイトモード**: 過去の対局URLを入力して相手の癖（攻撃型/守備型）を瞬時に学習

```bash
python gui.py
```

---

## 6. 友人配布用スタンドアロン EXE のビルド (`build_exe.py`)

Python や依存ライブラリがインストールされていない友人環境でもダブルクリックで実行可能な単一実行ファイル（`dist/MahjongAI_Assistant.exe`）を生成します。

```bash
# ワンクリックビルド（コンソール非表示・データ同梱）
python build_exe.py

# デバッグ用にコンソール画面を残す場合
python build_exe.py --console
```

- **生成物**: `dist/MahjongAI_Assistant.exe`（約34MB）
- **リソース解決**: `utils/path_helper.py` により、通常開発環境と PyInstaller 一時展開ディレクトリ（`_MEIPASS`）の双方が自動解決されます。

