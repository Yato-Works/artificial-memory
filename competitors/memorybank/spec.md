# MemoryBank System Specification & Audit

## 1. 概要
- **対象**: MemoryBank (Zhong et al., 2023)
- **論文**: "MemoryBank: Enhancing Large Language Models with Long-Term Memory"
- **アーキテクチャ分類**: Cognitive Forgetting Curve (Ebbinghaus)

---

## 2. 実装詳細

### (1) 忘却曲線メカニズム
- 各ターンにタイムスタンプを付与して保存。
- 検索時、経過時間 $\Delta t$ に基づきエビングハウス忘却曲線の保持度 $R$ を計算：
  $$R = e^{-\Delta t / S}$$
- スコア算出式：
  $$\text{Score} = \alpha \cdot \text{Similarity} + (1 - \alpha) \cdot R$$
  - $\alpha = 0.6$
  - $\text{decay\_factor} = 0.95$

### (2) 検索・コンテキスト
- 計算された合成スコアの上位 Top-k（k=10）を取得し、最大2,000トークンの枠内でLLMへ提供。
