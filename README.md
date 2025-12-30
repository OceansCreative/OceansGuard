# OceansGuard

OceansGuard は、生成AIによるコード変更を  
**CI・契約・セキュリティで機械的に裁くためのガードレール**です。

## 目的
- AIにコードを書かせても事故らせない
- 人が説明・確認・判断しなくてよい開発
- どの言語・フレームワークでも共通運用

## 基本思想
- AIは「提案者」
- 正しさは「テスト・契約・ポリシー」が決める
- 通らない変更は採用されない

## 使い方（各プロジェクト側）
```bash
python path/to/aiguard.py init
python path/to/aiguard.py pack
python path/to/aiguard.py check

対応フェーズ

開発前 / 開発途中 / 開発後 すべて対応


---

## ③ あなたの「不可がほぼ無い」運用フロー（確定）
**どの案件でもこれだけ**



AIに投げる前 → ai:pack
AI差分適用後 → ai:check
通ったら → 採用


- 考えない
- 説明しない
- レビューしない  

---

## ④ 最初のGit操作（推奨）
```bash
git add .
git commit -m "feat: initial OceansGuard core structure"
git tag v0.1.0
git push origin main --tags