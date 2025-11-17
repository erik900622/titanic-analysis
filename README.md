# Titanic Customer Behavior Analysis

這個專案使用 Kaggle 上的 Titanic 資料集，但我不是把它當成單純的機器學習練習題目，而是試著詮釋成「客戶行為 / 成功率預測」的商業分析專案。

在商業情境中，可以把：

- `Survived = 1` 視為：客戶成功完成某個關鍵行為  
  （例如：完成註冊、完成訂單、留在平台、不流失）
- `Survived = 0` 視為：客戶沒有成功  
  （例如：註冊失敗、訂單取消、流失）

目標是：  
> 找出哪些因子會影響「客戶成功」，並建立可解釋的預測模型與分群洞察。

---

## 1. 專案結構

目前檔案大致如下：

- `Data/`：放原始資料（train.csv, test.csv, submission.csv）
- `image/`：放視覺化圖片（生存率圖、特徵重要性等）
- `noteboooks/`：Jupyter Notebook（EDA、清洗、建模）
- `README.md`：本說明文件

之後會逐步整理成：

- 01_EDA.ipynb：探索性資料分析  
- 02_DataCleaning_FeatureEngineering.ipynb：缺失值處理與特徵工程  
- 03_Modeling_Logistic_RF.ipynb：邏輯迴歸 + 隨機森林建模與評估  
- 04_FinalPrediction.ipynb：在 test 資料集做預測並輸出 submission.csv

---

## 2. 分析流程（我做了什麼）

### (1) 資料理解 & EDA

- 檢查欄位意義、資料型態、缺失值分布  
- 觀察關鍵特徵與生存率的關係：
  - Sex vs Survived：女性生存率遠高於男性  
  - Pclass vs Survived：艙等越高，生存率越高  
  - Age vs Survived：小孩生存率較高  
  - FamilySize vs Survived：2–4 人小家庭生存率最高  
  - Embarked vs Survived：不同港口代表不同市場/階級結構

這些 EDA 不是只畫圖，而是用來回答：
> 「哪些變數真的有訊號？哪些更像是噪音？」

---

### (2) 資料清洗與特徵工程

- Age：以中位數補值  
- Fare：以中位數補值  
- Cabin：缺失過多，先全部捨棄  
- Embarked：用眾數補值  
- Sex：轉換成 0/1  
- Embarked：做 one-hot encoding（Embarked_Q, Embarked_S）  
- 新增 `FamilySize = SibSp + Parch + 1` 來表示家戶規模  

這些處理都以「業務意義」為前提，而不是硬套公式。

---

### (3) 建立模型

使用兩種模型：

1. **Logistic Regression**  
   - 當作 baseline  
   - 優點是係數可以解釋每個特徵對「成功機率」的影響方向與強度  

2. **Random Forest**  
   - 捕捉非線性與特徵交互作用  
   - 提供 feature importance  
   - Validation accuracy 約 0.82 左右  

同時觀察：

- Confusion matrix  
- Precision / Recall / F1-score  
- Type 1 / Type 2 error 在這類問題中的意義

---

### (4) 特徵重要性與洞察

從隨機森林的 feature importance 中可以看到：

- **Fare（票價）：代表客戶價值，愈願意付錢的客戶成功率愈高  
- **Sex（性別）：不同客群類型，行為差異明顯  
- **Age（年齡）：反映生命週期與行為特徵  
- **Pclass（艙等）：價值分級（類似 VIP / 一般客戶）  
- **FamilySize（家庭大小）：小家庭成功率最高，太大或單人都較差  

這些結果同時被 EDA 與模型驗證，並不是「黑箱」結果。

---

## 3. 商業化詮釋（為什麼跟實際工作有關）

雖然這是 Titanic 資料集，但整個流程可以直接套用在：

- 客戶流失預測（誰會留下來 / 誰會離開）  
- 訂單是否會被取消  
- 使用者是否會完成 onboarding  
- 客戶是否會轉換（conversion）  
- 風險客戶識別（high risk / low risk）

我在這個專案中練習的不是 Kaggle 排名，而是：

- 如何把問題轉成可以分析的形式  
- 如何根據欄位意義設計特徵  
- 如何用圖表與指標說明「為什麼」  
- 如何用簡單、可解釋的模型支援決策  
- 如何從預測結果抽出對業務有用的分群與建議  
