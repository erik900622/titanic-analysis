# 🛳 Titanic 生存分析：從使用者故事出發，建立可支撐決策的資料洞察

本專案以 Kaggle Titanic 資料為基礎，但分析目標並非預測生存，而是回答一個更本質的問題：

> **「在資訊受限的環境下，哪些因素最能解釋『成功率差異』？」**

我將此題視為一個典型的 decision problem：  
**有限資料 → 需要快速判斷 → 找出關鍵變數 → 建立簡潔可靠的 decision rule。**

透過 User Story → EDA → Feature Engineering → Model Validation，  
我萃取出四個高影響力的決定因子（專案的最後有Data visualization佐證）：

1. **價值階層（Pclass / Fare）**：代表資源取得優勢。  
2. **族群特徵（Sex / Age）**：反映社會規範與生命週期行為。  
3. **家庭結構（FamilySize）**：影響行動協調與決策效率。  
4. **市場差異（Embarked）**：象徵不同 socio-economic segment。

整體目標是建立一個 **能解釋行為差異與成功率**  
的邏輯框架，而不是追求模型分數。

---

# 結論

雖然 Titanic 是乾淨的資料集，資料清洗的部分無法多著墨，但我認為分析方法對現代商業場景具有高度可轉移性。

## **1｜建立可解釋的成功率模型（Conversion / Retention Model）**
Survived = 1 or 0 可對應幾個常見的問題：

- 新客 onboarding 是否完成  
- 付款是否成功  
- 用戶是否留存  
- 訂單是否取消  

本專案展示我能從資料中建立清晰且可解釋的成功率模型。

---

## **2｜識別成功率的關鍵驅動因子（Key Driver Analysis）**
Sex, Pclass, Fare, Age, FamilySize 對應到企業常見概念：

- 客戶價值（Value Tier）  
- 族群差異（Cohort Behavior）  
- 行為動機（Lifecycle Behavior）  
- 團體/家庭行為（Group-level Decision）  

這可提供：

- 分群策略（Segmentation）  
- 產品定位建議  
- 流失預測（Churn Risk）  
- 優先支援策略（Priority Rules）

---


# 👤 User Story：建立情境與決策需求

假設你是船上負責緊急應變的主管。  
船艙進水、救生艇不足、資訊混亂，你需要在短時間內回答：

> **1. 哪些乘客生存率高（可自行應對）？  
> 2. 哪些乘客生存率低（需要優先支援）？  
> 3. 這些判斷可依據哪些客觀資料？**

手上唯一的資訊來自乘客名單：

- 艙等（價值階層）  
- 年齡、性別（人口特徵）  
- 家庭人數（協同行為）  
- 票價（支付能力）  
- 登船地（市場來源）

你的思考流程必須從：

> **「這些變數是否提供足夠訊號，能協助我做出穩定判斷？」**

這也就是本專案的核心：  
**用現有資訊，建立足以支撐決策的邏輯。**

---

# 📌 問題定義

我把問題拆分成三個部分：
### **Q1｜哪些變數具有解釋力？**
（找到生存率差異最大的 driver）

### **Q2｜變數之間是否存在交互效應？**
（例如：艙等 × 性別 → 不同族群的行為呈現）

### **Q3｜能否形成一套簡潔、可重複使用的 decision rule？**


---

# 🔍 分析方法（Structured Approach）

## 1️⃣ 假設建立（Hypothesis-Driven）
先定義方向：

- Pclass/Fare → 代表資源與支付能力  
- Sex/Age → 社會規範、生命週期  
- FamilySize → 協同行為能力  
- Embarked → 背景差異、segment 來源  

這提供 EDA 清楚的分析 lens。

---

## 2️⃣ EDA：驗證訊號強度（Signal Identification）

EDA 的目的不是視覺化，而是判斷變數是否值得建模。

主要觀察如下：

- 女性生存率顯著高 → **族群差異明顯**  
- 艙等與票價高者成功率高 → **價值分層**  
- 小家庭（2–4 人）成功率高 → **協同行為有效率**  
- 港口差異反映 segment 行為 → **市場差異存在**

結論：  
這些變數具有足夠訊號，值得納入模型。

---

## 3️⃣ 特徵工程（Feature Engineering）

所有變數均對應到可解釋的商業概念：

- `FamilySize = SibSp + Parch + 1` → 協同行為  
- `Fare` → 資源/支付能力  
- `Age` → 生命週期  
- `Pclass` → 階層差異  
- 類別欄位 one-hot → 避免強制排序錯誤  

這確保模型建立在行為邏輯，而非純計算。

---

## 4️⃣ 模型驗證（Model Validation）

採用兩種模型，目的不同：

- **Logistic Regression** → 驗證方向（正向/負向）  
- **Random Forest** → 驗證特徵重要性（non-linear + interactions）

結果一致：

- Sex、Pclass、Fare、Age、FamilySize 為最強 driver  
- Embarked 為 segmentation indicator

模型不是核心，  
關鍵在於 **模型結果是否支持洞察**。


## **3｜形成簡單、可操作的 Decision Rule**
根據分析，可建立如：

- 高價值 + 女性 + 小家庭 → 高成功率 → 可自行完成流程  
- 男性 + 低艙等 + 單人乘客 → 高風險 → 需要額外支援  


##  專案結構

目前檔案大致如下：

- `Data/`：放原始資料（train.csv, test.csv, submission.csv）
- `image/`：放視覺化圖片（生存率圖、特徵重要性等）
- `notebooks/`：Jupyter Notebook（EDA、清洗、建模）
- `README.md`：本說明文件

之後會逐步整理成：

- 01_EDA.ipynb：探索性資料分析  
- 02_DataCleaning_FeatureEngineering.ipynb：缺失值處理與特徵工程  
- 03_Modeling_Logistic_RF.ipynb：邏輯迴歸 + 隨機森林建模與評估  
- 04_FinalPrediction.ipynb：在 test 資料集做預測並輸出 submission.csv

---

##  分析流程（我做了什麼）

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

- Fare（票價）：代表客戶價值，愈願意付錢的客戶成功率愈高  
- Sex（性別）：不同客群類型，行為差異明顯  
- Age（年齡）：反映生命週期與行為特徵  
- Pclass（艙等）：價值分級（類似 VIP / 一般客戶）  
- FamilySize（家庭大小）：小家庭成功率最高，太大或單人都較差  

這些結果同時被 EDA 與模型驗證，並不是「黑箱」結果。

---


## 📊 Key Visualizations & Insights

---

### 1. Age Distribution
<img src="titanic/images/age_distribution.png" width="500">

Titanic 乘客年齡呈現右偏分布，20–40 歲為最主要族群。

---

### 2. Age vs Fare vs Pclass
<img src="titanic/images/age_fare_pclass.png" width="500">

高票價（Fare）乘客多屬於高艙等（Pclass=1），顯示支付能力與艙等呈明顯關聯。

---

### 3. Survival Rate by Age & Sex
<img src="titanic/images/age_sex.png" width="500">

女性在不同年齡層中普遍生存率較高，而男性年紀越大生存率下降越明顯。

---

### 4. Survival Rate by Class and Sex
<img src="titanic/images/class_sex_survival.png" width="500">

1 等艙女性生存率接近 100%，艙等（Pclass）與性別（Sex）是最強的生存預測因子。

---

### 5. Survival Rate by Family Size
<img src="titanic/images/familySize_survived.png" width="500">

小家庭（FamilySize 2–4）有最高生存率；單人或大家庭的生存率較低。

---

### 6. Survival Rate by Class and Sex (Seaborn)
<img src="titanic/images/survival_class_sex.png" width="500">

與 plotly 版本一致，艙等愈低生存率愈差，女性在各艙等皆有較高生存機會。

---

### 7. Survival Rate by Family Size (Seaborn)
<img src="titanic/images/survival_familysize.png" width="500">

再次驗證 FamilySize 的非線性關係：小家庭優於單人與大家庭。

---

### 8. Feature Importance (Logistic Regression)
<img src="titanic/images/feature.png" width="350">

Sex、Pclass、Fare、Age 為最關鍵特徵，模型能有效擷取人類可理解的模式。
