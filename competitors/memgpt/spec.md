# MemGPT / Letta System Specification & Audit

## 1. 概要
- **対象**: MemGPT / Letta (Packer et al., 2023)
- **論文**: "MemGPT: Towards LLMs as Operating Systems"
- **アーキテクチャ分類**: Hierarchical OS-like Memory (Working Context + Archival Memory)

---

## 2. 実装詳細

### (1) メモリ階層
1. **Working Context (Core Memory)**:
   - 直近の対話ターンを保持するスライディングFIFOバッファ（最大1,000トークン）。
2. **Archival Memory**:
   - 過去の全対話ターンをベクトル化して格納する外部記憶ストア（最大1,000トークン）。

### (2) 読み出し・コンテキスト構築
- 質問が来ると、まず Working Context（直近の文脈）を確実にコンテキストに配置。
- 残りのトークン予算（約1,000トークン）を用いて、Archival Memory からコサイン類似度上位のターン（Top-k=10）を検索して充填。
- 合計2,000トークンの枠内でLLMへ渡す。
