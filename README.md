# 🛳 Titanic 生存分析：在壓力情境下建立可支撐決策的資料框架

這個專案不是在「猜誰會活下來」，而是在回答一個更接近真實決策的問題：

> **在資訊有限、時間緊迫、資源不足的情況下：  
> 我要如何用資料判斷「誰必須優先救援」，誰的成功率低到可能被迫放棄？**

我把 Titanic 當成一個 decision problem，而不是純 machine learning problem：

- 決策成本不對稱（誤救 vs 誤放棄）  
- 目標是 **建立簡潔穩定的 decision rule** 

透過 User Story → EDA → Feature Engineering → Model Validation，  
我最後找出四個最關鍵的決策因子：

1. **價值階層（Pclass / Fare）**  
2. **族群特徵（Sex / Age）**  
3. **家庭結構（FamilySize）**  
4. **市場差異（Embarked）**

這套邏輯可以直接遷移到實際商業場景，例如：轉換率、留存率、風險評分。

---

## 👤 User Story：決策場景與錯誤成本

情境設定：

- 船艙進水，救生艇數量有限  
- 你是負責決策的主管  
- 你必須在極短時間內決定：**誰一定要救？誰可以等？誰很可能來不及？**

你僅能使用的資訊：

- 艙等（Pclass）  
- 性別、年齡（Sex, Age）  
- 家庭大小（FamilySize）  
- 票價（Fare）  
- 登船港口（Embarked）

這裡的分類錯誤成本高度不對稱：

- **Type I Error（False Positive｜誤救）**  
  把「其實能自己活下來的人」標成高風險，多分了一個救生艇位子。  
  → 成本：浪費資源，但沒有人因此喪命。

- **Type II Error（False Negative｜誤放棄）**  
  把「其實會死亡的人」判為安全，結果沒有被救。  
  → 成本：直接導致死亡，是最不能接受的錯誤。

因此這個問題的本質是：

> **如何在有限資訊下，最大限度降低 Type II Error。**

---

## ⚠️ Error Trade-off：決策樹視覺化

```mermaid
flowchart TD

    A["模型判斷：是否需要救援？"] --> B1["預測：需要救援"]
    A --> B2["預測：不需要救援"]

    B1 --> C1["實際：高風險 → True Positive（救到該救的人）"]
    B1 --> C2["實際：低風險 → Type I Error（誤救，浪費救生艇）"]

    B2 --> C3["實際：低風險 → True Negative（正確放行）"]
    B2 --> C4["實際：高風險 → Type II Error（誤放棄，可能死亡）"]

# 📌 問題定義

將決策邏輯拆成三個明確問題：

### **Q1｜哪些變數對成功率（生存）影響最大？**
→ 找出高風險 / 低風險的主要 driver。

### **Q2｜是否存在明顯的交互效應？**
→ 例如艙等 × 性別是否形成不同風險族群。

### **Q3｜能否形成一條簡潔、可重複使用的 decision rule？**
→ 壓力情境下也能快速採用。

---

# 🎯 三個問題的答案

### **A1｜關鍵變數包括 Sex、Pclass、Fare、Age、FamilySize。**
它們持續在 EDA、Logistic Regression、Random Forest 中呈現高解釋力。

### **A2｜最關鍵的交互效應是「艙等 × 性別」。**
- 1 等艙女性 → 幾乎全部生存（非常低風險）  
- 3 等艙男性 → 生存率最低（高度高風險）  

其次是：
- Age × FamilySize：有小孩的小家庭成功率明顯較高。

### **A3｜可以用上述 driver 建構出可落地的 decision rule。**

> **高艙等＋女性/兒童＋小家庭 → 可自行應對**  
> **低艙等＋成年男性＋單人/大家庭 → 高風險，需優先救援**

這條規則簡潔、可重複、可擴展到其他商業場景（如流失風險判斷）。

---

# 📊 支撐洞察的圖表



### 1. Age Distribution  
<img src="titanic/images/age_distribution.png" width="500">

### 2. Age vs Fare vs Pclass  
<img src="titanic/images/age_fare_pclass.png" width="500">

### 3. Survival Rate by Age & Sex  
<img src="titanic/images/age_sex.png" width="500">

### 4. Survival Rate by Class and Sex  
<img src="titanic/images/class_sex_survival.png" width="500">

### 5. Survival Rate by Family Size  
<img src="titanic/images/familySize_survived.png" width="500">

### 6. Survival Rate by Class and Sex (Seaborn)  
<img src="titanic/images/survival_class_sex.png" width="500">

### 7. Survival Rate by Family Size (Seaborn)  
<img src="titanic/images/survival_familysize.png" width="500">

### 8. Feature Importance  
<img src="titanic/images/feature.png" width="350">

---

# 🔍 分析方法

這部分展示我在解決問題時的結構化思考，而不是把 Titanic 當 Kaggle 題目在做。

## 1️⃣ 建立假設

資料還沒看之前先假設：

- 高艙等與高票價 = 資源取得優勢  
- 女性、小孩 = 社會規範優先照顧  
- 小家庭 = 協同行為效率較高  
- 登船地差異 = 不同 socio-economic segment  

這讓 EDA 有方向。

---

## 2️⃣ EDA：辨識「真的有訊號」的變數

使用描述統計 + 可視覺化檢查每個特徵：

- Sex：女性生存率遠高於男性  
- Pclass/Fare：價格與階層差異非常明顯  
- Age：兒童成功率高，年齡越大越不利（男性尤其顯著）  
- FamilySize：2–4 人的小家庭成功率最高  
- Embarked：不同港口代表不同市場結構  

結論：這些變數具有穩定訊號，值得做成特徵。

---

## 3️⃣ Feature Engineering

把邏輯概念具象化成模型可用的特徵：

- `FamilySize = SibSp + Parch + 1`  
- Sex → 0/1  
- Embarked → one-hot encoding（避免錯誤排序）  
- Age, Fare → 以中位數補缺失  
- Cabin → 缺失過高，捨棄  

這一步確保模型建立在邏輯，而不只是跑演算法。

---

## 4️⃣ Model Validation（Logistic Regression + Random Forest）

使用兩種模型交叉驗證洞察：

### ✔ Logistic Regression  
確認各特徵的影響方向（正向、負向），並檢查是否符合行為邏輯。

### ✔ Random Forest  
確認特徵重要性排名，檢查非線性特徵與交互效果。

兩者結果一致：  
Sex、Pclass、Fare、Age、FamilySize 是真正的主因子。

---


## 商業價值

### 1｜成功率模型可以直接套用在商業情境

`Survived = 1/0` 可類比為：

- 付款是否成功  
- 新用戶 onboarding 是否完成  
- 客戶是否留存／流失  
- 訂單是否成功／被取消  

重點不在 Titanic 本身，而是：

> **從一個二元結果中，拆出「誰成功、誰失敗」背後的結構。**

---

### 2｜成功率有清楚的 Key Drivers

分析結果顯示，生存率主要由以下變數驅動：

- Sex（性別）  
- Pclass（艙等）  
- Fare（票價）  
- Age（年齡）  
- FamilySize（家庭大小）  

它們對應到常見的商業概念：

| 變數             | 商業對應概念                 |
|------------------|------------------------------|
| Sex              | 不同 cohort 行為差異         |
| Pclass / Fare    | 客戶價值分層（Value Tier）   |
| Age              | 生命週期行為（Lifecycle）    |
| FamilySize       | 群體決策與協同行為           |
| Embarked         | 市場／背景差異（Segment）    |

---

### 3｜可以形成可落地的 Decision Rule

綜合 EDA 與模型結果，可以用一句話描述決策規則：

> **低艙等（Pclass=3）＋ 成年男性 ＋ 單人或大家庭 ＋ 低票價 → 最高風險，應優先救援。**  
> **高艙等＋ 女性或兒童 ＋ 小家庭（2–4 人）＋ 較高票價 → 相對安全，可較晚處理。**

這條 rule：

- 簡單到可以在壓力情境下使用  
- 能明確降低「錯放棄」的風險（Type II Error）  
- 可以遷移到各種「有限資源、決策成本不對稱」的場景  

---




