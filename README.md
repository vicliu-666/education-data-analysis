# 教育數據分析：基於機器學習的學生學習風格分群與成績預測
# (Educational Data Analysis: Student Profiling and Grade Prediction)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 專案簡介 (Project Overview)

本專案為 **「機器學習在教育上的應用」** 期末專題報告。

傳統教育現場常面臨無法即時識別落後學生的問題。本研究旨在利用數據科學方法，建立一套**「早期預警系統」**。透過分析學生的學習行為（如出勤率、作業完成度、線上互動），我們試圖在不依賴期中/期末考分數的情況下，預測學生的最終成績表現，並識別出不同類型的學習者畫像，以利教育者進行適性化介入。

### 核心目標
1.  **監督式學習**：建立預測模型，依據平時行為預測期末成績等級 (Final Grade)。
2.  **非監督式學習**：透過 K-Means 進行學生分群，識別具備可解釋性的學習者輪廓 (Profiles)。

---

## 📂 資料集來源 (Dataset)

本研究使用整合自 Kaggle 與 Zenodo 的教育數據集，經過去識別化處理。

* **資料來源**：[Zenodo - Student Performance and Learning Behavior Dataset](https://zenodo.org/records/16459132)
* **資料規模**：14,003 筆學生紀錄，包含 16 個特徵欄位。
* **特徵類別**：
    * **學習投入**：`StudyHours`, `Attendance`, `AssignmentCompletion`, `OnlineCourses`
    * **學習環境**：`Resources`, `Internet`, `EduTech`
    * **心理因素**：`Motivation`, `StressLevel`
    * **學業成效**：`ExamScore` (考試分數), `FinalGrade` (期末成績 - Target)

---

## 🚀 分析方法 (Methodology)

### 1. 資料前處理 (Data Preprocessing)
* 缺失值處理與異常值檢測。
* 類別特徵編碼 (Label Encoding)。
* 特徵縮放 (StandardScaler) 以適用於 K-Means 分群。

### 2. 監督式學習 (Supervised Learning)
* **任務**：預測 `FinalGrade` (分類任務)。
* **模型**：Random Forest Classifier。
* **關鍵策略**：**防止資料洩漏 (Data Leakage Prevention)**。
    * 為了模擬「學期中」的預警情境，我們在訓練模型時**移除了 `ExamScore`** 欄位。雖然這稍微降低了準確率，但證明了僅靠「作業」與「出席」等行為數據，即可有效預測成績。
* **評估指標**：Accuracy, Macro-F1, ROC-AUC。

### 3. 非監督式學習 (Unsupervised Learning)
* **任務**：學習者分群 (Student Segmentation)。
* **模型**：K-Means Clustering。
* **策略**：僅使用「行為」與「心理」特徵進行分群（不含成績），再透過事後驗證 (Post-hoc Validation) 觀察各群集在成績上的差異。
* **最佳群數**：$k=3$ (基於 Elbow Method 與 Silhouette Score)。

---

## 📊 實驗結果 (Results)

### 監督式學習表現
* **準確率 (Accuracy)**：**92.3%**
* **關鍵特徵**：特徵重要度分析顯示，`AssignmentCompletion` (作業完成度) 與 `Attendance` (出席率) 是影響成績最重要的預測因子。

### 非監督式學習洞察
成功識別出三種學習者畫像：
1.  **高投入型**：高出席、高作業完成率，對應較高的平均成績。
2.  **低動機/高風險型**：低互動、高壓力，成績分佈明顯偏低。
3.  **一般型**：各項指標居中。
> *Post-hoc 分析證實，雖然分群過程未參考成績，但分出的群體在最終成績上具有顯著差異。*

---

## 🛠️ 安裝與執行 (Installation & Usage)

### 專案結構