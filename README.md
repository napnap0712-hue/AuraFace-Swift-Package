# AuraFaceKit

Swift Playgrounds / iOS 17+ 向けのAuraFace Core MLラッパーです。

## 構成

```text
Package.swift
Sources/
└── AuraFaceKit/
    └── AuraFaceManager.swift
```

## モデル

- Source: `RuiSumida/AuraFace-v1-CoreML`
- Base model: `fal/AuraFace-v1`
- License: Apache-2.0
- File: `FaceEmbedding.mlmodel` (~130 MB)
- Input: `faceImage` — 112×112 RGB、5点アラインメント推奨
- Output: `embedding` — 512 dimensions
- SHA-256: `9cb10bef2141a36619bb1fdbf1e0e14e2519c6da4b7e9b9969a4d67702d7122b`

## 動作

初回だけモデルをダウンロードし、SHA-256を検証して端末内でCore MLへコンパイルします。
2回目以降は端末内に保存したコンパイル済みモデルを使います。

**園児写真を外部サーバーへ送信する処理はありません。**
外部通信はモデル本体の初回取得だけです。

## Swift Playgrounds側

パッケージを追加したら：

```swift
import AuraFaceKit

try await AuraFaceManager.shared.prepare()
let embedding = try await AuraFaceManager.shared.generateEmbedding(for: faceImage)
```

`faceImage`には、Vision等で切り出して可能なら5点アラインメントした顔画像を渡します。
