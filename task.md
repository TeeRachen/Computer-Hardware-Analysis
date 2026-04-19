# Project: PC Hardware Market Analysis & Recommendation Engine

## Overview
Develop a Python-based software application (`app.py`) that imports a suite of PC hardware datasets to perform Exploratory Data Analysis (EDA), deploy a Predictive Pricing Model, and run an interactive PC Part Picker and Price Appraiser via a Command-Line Interface (CLI).

---

## 1. Datasets
Location: `/Dataset/`
* `CaseData.csv`, `CPUCoolerData.csv`, `CPUData.csv`, `GPUData.csv`, `HDDData.csv`, `MonitorData.csv`, `MotherboardData.csv`, `PSUData.csv`, `RAMData.csv`, `SSDData.csv`

---

## 2. Data Preprocessing Requirements
Before executing features, the script must parse and clean the raw data:
1. **Price Columns:** Remove `$` and `,` symbols; cast to float.
2. **Spec Columns:** Strip string units (e.g., `MHz`, `GB`, `W`) from columns like `Boost Clock`, `Vram`, and `TDP`; cast to numeric types.
3. **Missing Data:** Drop or impute rows missing critical pricing or performance data.

---

## 3. Feature 1: Advanced Exploratory Data Analysis (EDA)
**Objective:** Generate statistical charts to identify the Pareto Front (best value), analyze market distribution, and discover feature correlations. 

**Specific Plots & Analytics Required:**
1. **Feature Correlation (Heatmap):** * Calculate a correlation matrix for numerical features (e.g., Price, VRAM, Clock, TDP).
    * Plot using a Heatmap with data annotations enabled to show exact correlation coefficients (e.g., 0.85).
    * **Insight:** Identify which specific hardware specifications have the strongest impact on the final retail price.
2. **GPU & CPU Value Matrices (Jointplot/Scatter):** * Plot `Price` vs. `Performance Score` (e.g., Core count or Clock x VRAM). 
    * **Coloring:** Group by `Producer` (AMD, Intel, NVIDIA).
    * **Styling:** Use high transparency (alpha) to reveal density in crowded budget tiers.
3. **RAM Sub-System (Boxplot):** * Plot `Price` vs. `Capacity (Size)`.
    * **Coloring:** Group by `Ram Type` (DDR4 vs. DDR5).
    * **Insight:** Analyze graph skew direction to identify exponential price premiums for higher capacities.
4. **Storage Sub-System (Boxplot & Line Plot):**
    * **Plot A (Consistency):** Boxplot of `Price` vs. `Size`, grouped by `Protocol` (NVMe vs. SATA).
    * **Plot B (Efficiency):** Line plot of `Price-per-GB` vs. `Size`.
    * **Insight:** Use the boxplot to show tier pricing, and the line plot to find the exact capacity where economies of scale peak (the cheapest cost-per-gigabyte).
5. **Power Supply (Bar Chart with Error Bars):**
    * Plot `Average Price` vs. `Efficiency Rating`.
    * **Insight:** Identify the "efficiency tax" required to jump from Bronze to Gold to Titanium.

**Output & File Organization:** Export all generated charts as high-resolution `.png` files, categorized into specific subdirectories within the main `/outputs/` folder. The script must automatically create these directories if they do not exist:
* `/outputs/Correlation_Matrices/` (Store the Heatmaps here).
* `/outputs/Value_Matrices/` (Store the GPU and CPU Price vs. Performance scatter plots here).
* `/outputs/Pricing_Tiers/` (Store the RAM and Storage Price vs. Capacity plots here).
* `/outputs/Efficiency_Analysis/` (Store the PSU Price vs. Efficiency charts here).

---

## 4. Feature 2: Predictive Modeling (Machine Learning)
**Objective:** Implement a Random Forest Regressor to predict the "Fair Market Price" of components based on their hardware specifications.

**Phase A: Training & Evaluation**
* Split data into training/testing sets (80/20). 
* Train the model to predict `Price` using numerical features (e.g., `Vram`, `Boost Clock`, `TDP`).
* Calculate and print the Mean Absolute Error (MAE) and R-squared score to the console. 
* Extract and export a horizontal bar chart of the "Feature Importances" to show which specs drive market pricing the most.

**Phase B: Interactive Price Appraiser (Inference)**
* Integrate the trained model into the CLI menu.
* Prompt the user to input custom specifications for a hypothetical or real-world component (e.g., "Enter the VRAM in GB:", "Enter the Boost Clock in MHz:").
* Feed these inputs into the trained model and print the predicted "Fair Market Price" to the console.

---

## 5. Feature 3: Interactive Command-Line Interface (CLI)
**Objective:** Create an interactive main menu that allows the user to navigate between the application's core features.

**Main Menu Flow:**
> **Welcome to the PC Hardware Tool**
> [1] Run Market Analysis (Generate EDA Charts)
> [2] Build a PC (Recommendation Engine)
> [3] Appraise a Component (ML Price Predictor)
> *Select an option:*

**Option [2] Recommendation System (PC Part Picker) Logic:**
1. **Input 1 (Budget):** Prompt: *"Enter your total budget in USD:"* (Must validate for numeric input).
2. **Input 2 (Use Case):** Prompt: *"Select primary use case: [1] Gaming [2] Video Editing [3] Home Office [4] AI Developer [5] Home Server:"*

**Compatibility Constraints (Strict Rules):**
1. CPU `Socket` == Motherboard `Socket`.
2. RAM `Ram Type` == Motherboard `Memory Type`.
3. PSU `Watt` >= (CPU `TDP` + GPU `TDP`) * 1.2 (20% overhead).

**Budget Allocation Algorithms:**
* **[1] Gaming:** Heavily weight the GPU (~45% of total budget), standard CPU, 16GB/32GB RAM.
* **[2] Video Editing:** Heavily weight the CPU (high core/thread count) and RAM (32GB+), standard GPU.
* **[3] Home Office:** Skip discrete GPU (filter for CPU with `Integrated GPU`), 16GB RAM, fast 1TB SSD, prioritize low total cost well under budget.
* **[4] AI Developer:** Prioritize GPU specifically by `VRAM` capacity (16GB+ minimum), ensure System RAM is 2x GPU VRAM, PSU requires 30% overhead.
* **[5] Home Server (NAS):** Allocate ~40% budget to high-capacity HDDs, prioritize Motherboards with high `SATA` port counts and Cases with high `Disk 3.5"` bays. Lowest possible TDP CPU.

**Output:** Print a formatted terminal receipt listing the selected components, individual prices, total cost, and remaining budget.