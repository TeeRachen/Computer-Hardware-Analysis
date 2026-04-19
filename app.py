import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ---------------------------------------------------------
# 1. Setup & Directory Management
# ---------------------------------------------------------
sns.set_theme(style="darkgrid", palette="muted")

OUTPUT_DIR = 'outputs'
SUB_DIRS = {
    'CORR': os.path.join(OUTPUT_DIR, 'Correlation_Matrices'),
    'VALUE': os.path.join(OUTPUT_DIR, 'Value_Matrices'),
    'PRICING': os.path.join(OUTPUT_DIR, 'Pricing_Tiers'),
    'EFFICIENCY': os.path.join(OUTPUT_DIR, 'Efficiency_Analysis')
}

for path in [OUTPUT_DIR] + list(SUB_DIRS.values()):
    os.makedirs(path, exist_ok=True)

# ---------------------------------------------------------
# 2. Data Loading & Master Cleaning
# ---------------------------------------------------------
def clean_price(df):
    if 'Price' in df.columns:
        df['Price'] = df['Price'].astype(str)
        df['Price'] = df['Price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.replace(' USD', '', regex=False)
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    return df

def extract_numeric(series, pattern):
    return series.astype(str).str.extract(pattern).astype(float)

def parse_size(size_str):
    if pd.isna(size_str):
        return None
    pieces = str(size_str).split()
    try:
        num = float(pieces[0])
        if len(pieces) > 1 and 'TB' in pieces[1].upper():
            return num * 1000
        return num
    except Exception:
        return None

print("Loading datasets...")
case_df = pd.read_csv('Dataset/CaseData.csv')
cpu_cooler_df = pd.read_csv('Dataset/CPUCoolerData.csv')
cpu_df = pd.read_csv('Dataset/CPUData.csv')
gpu_df = pd.read_csv('Dataset/GPUData.csv')
hdd_df = pd.read_csv('Dataset/HDDData.csv')
monitor_df = pd.read_csv('Dataset/MonitorData.csv')
motherboard_df = pd.read_csv('Dataset/MotherboardData.csv')
psu_df = pd.read_csv('Dataset/PSUData.csv')
ram_df = pd.read_csv('Dataset/RAMData.csv')
ssd_df = pd.read_csv('Dataset/SSDData.csv')


def preprocess_data():
    global gpu_df, cpu_df, monitor_df, psu_df, ram_df, ssd_df, hdd_df, motherboard_df, case_df

    dfs = [gpu_df, cpu_df, monitor_df, psu_df, ram_df, ssd_df, hdd_df, motherboard_df, case_df]
    for i in range(len(dfs)):
        dfs[i] = clean_price(dfs[i])
    gpu_df, cpu_df, monitor_df, psu_df, ram_df, ssd_df, hdd_df, motherboard_df, case_df = dfs

    gpu_df['Vram_GB'] = extract_numeric(gpu_df['Vram'], r'(\d+)')
    gpu_df['Boost_Clock_MHz'] = extract_numeric(gpu_df['Boost Clock'], r'(\d+)')
    gpu_df['TDP_W'] = extract_numeric(gpu_df['TDP'], r'(\d+)')
    gpu_df['Perf_Score'] = (gpu_df['Boost_Clock_MHz'] * gpu_df['Vram_GB']) / 1000

    cpu_df['Cores'] = pd.to_numeric(cpu_df['Cores'], errors='coerce')
    cpu_df['Threads'] = pd.to_numeric(cpu_df['Threads'], errors='coerce')
    cpu_df['TDP_W'] = extract_numeric(cpu_df['TDP'], r'(\d+)')
    cpu_df['Base_Clock_GHz'] = extract_numeric(cpu_df['Base Clock'], r'(\d+\.?\d*)')
    cpu_df['Integrated_GPU'] = cpu_df['Integrated GPU'].astype(str).str.lower().isin(['true', 'yes'])

    monitor_df['Refresh_Rate_Hz'] = extract_numeric(monitor_df['Refresh Rate'], r'(\d+)')
    psu_df['Watt_W'] = extract_numeric(psu_df['Watt'], r'(\d+)')
    ram_df['Size_GB'] = extract_numeric(ram_df['Size'], r'(\d+)')
    ram_df['Ram_Type'] = ram_df['Ram Type'].astype(str).str.extract(r'(DDR\d+)')

    ssd_df['Size_GB'] = ssd_df['Size'].apply(parse_size)
    ssd_df['Protocol'] = ssd_df['Protocol'].replace({'NVM': 'NVMe'}).fillna('Unknown')
    hdd_df['Size_GB'] = hdd_df['Size'].apply(parse_size)
    motherboard_df['SATA_Count'] = extract_numeric(motherboard_df['SATA'].astype(str), r'(\d+)')

    disk_cols = case_df.filter(like='Disk 3.5').columns
    case_df['Disk_3_5_Bays'] = extract_numeric(case_df[disk_cols[0]].astype(str), r'(\d+)') if len(disk_cols) > 0 else 0

    gpu_df.dropna(subset=['Price', 'Vram_GB', 'Perf_Score'], inplace=True)
    cpu_df.dropna(subset=['Price', 'Cores', 'Base_Clock_GHz'], inplace=True)
    ram_df.dropna(subset=['Price', 'Size_GB', 'Ram_Type'], inplace=True)

# ---------------------------------------------------------
# 3. Feature 1: Advanced EDA Charts
# ---------------------------------------------------------
def plot_correlation_heatmap(df, columns, title, filename):
    df_plot = df[columns].dropna()
    if df_plot.empty:
        print(f"Skipping {title}: no numeric data")
        return
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_plot.corr(), annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(SUB_DIRS['CORR'], filename), dpi=300)
    plt.close()
    print(f"Saved {os.path.join(SUB_DIRS['CORR'], filename)}")


def plot_price_per_gb(df, size_col, price_col, group_col, title, filename):
    df_plot = df.dropna(subset=[size_col, price_col, group_col])
    df_plot = df_plot[df_plot[size_col] > 0].copy()
    if df_plot.empty:
        print(f"Skipping {title}: no valid size data")
        return
    df_plot['Price_per_GB'] = df_plot[price_col] / df_plot[size_col]
    plt.figure(figsize=(10, 6))
    for group, subset in df_plot.groupby(group_col):
        bucketed = subset.groupby(size_col)['Price_per_GB'].mean().sort_index()
        if bucketed.empty:
            continue
        plt.plot(bucketed.index, bucketed.values, marker='o', label=str(group))
    plt.title(title)
    plt.xlabel('Size (GB)')
    plt.ylabel('Price per GB')
    plt.legend(title=group_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(SUB_DIRS['PRICING'], filename), dpi=300)
    plt.close()
    print(f"Saved {os.path.join(SUB_DIRS['PRICING'], filename)}")


def generate_eda_charts():
    print("\nGenerating categorized EDA charts...")
    
    # ---------------------------------------------------------
    # 1. Feature Correlation (Heatmaps)
    # ---------------------------------------------------------
    # GPU Correlation
    plt.figure(figsize=(10, 8))
    gpu_corr_cols = ['Price', 'Vram_GB', 'Boost_Clock_MHz', 'TDP_W', 'Perf_Score']
    sns.heatmap(gpu_df[gpu_corr_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('GPU Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SUB_DIRS['CORR'], 'gpu_correlation.png'), dpi=300)
    plt.close()

    # CPU Correlation
    plt.figure(figsize=(10, 8))
    cpu_corr_cols = ['Price', 'Cores', 'Threads', 'TDP_W', 'Base_Clock_GHz']
    sns.heatmap(cpu_df[cpu_corr_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('CPU Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SUB_DIRS['CORR'], 'cpu_correlation.png'), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 2. Value Matrices (Market Distribution)
    # ---------------------------------------------------------
    # GPU: Price vs VRAM (Professional Jointplot)
    g_gpu = sns.jointplot(
        data=gpu_df, 
        x='Vram_GB', 
        y='Price', 
        hue='Producer', 
        alpha=0.5,      # Reveals density in crowded budget tiers
        s=60,           # Standardized dot size
        edgecolor='w',  # White edges help dots stand out
        linewidth=0.5
    )
    g_gpu.ax_joint.set_xlabel('VRAM Capacity (GB)', fontweight='bold')
    g_gpu.ax_joint.set_ylabel('Market Price (USD)', fontweight='bold')
    g_gpu.fig.suptitle('GPU Market Analysis: Price vs. VRAM', y=1.02, fontsize=14, fontweight='bold')
    g_gpu.fig.savefig(os.path.join(SUB_DIRS['VALUE'], 'gpu_value_matrix.png'), dpi=300)
    plt.close('all')

    # CPU: Price vs Cores
    g_cpu = sns.jointplot(
        data=cpu_df, 
        x='Cores', 
        y='Price', 
        hue='Producer', 
        alpha=0.5
    )
    g_cpu.ax_joint.set_xlabel('Core Count', fontweight='bold')
    g_cpu.ax_joint.set_ylabel('Market Price (USD)', fontweight='bold')
    g_cpu.fig.suptitle('CPU Market Analysis: Price vs. Core Count', y=1.02, fontsize=14, fontweight='bold')
    g_cpu.fig.savefig(os.path.join(SUB_DIRS['VALUE'], 'cpu_value_matrix.png'), dpi=300)
    plt.close('all')

    # ---------------------------------------------------------
    # 3. Pricing Tiers (Boxplots)
    # ---------------------------------------------------------
    # RAM Pricing Tiers
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=ram_df, x='Size_GB', y='Price', hue='Ram_Type')
    plt.title('RAM Pricing Tiers: Capacity vs. Market Price', fontsize=14, fontweight='bold')
    plt.xlabel('Total Capacity (GB)')
    plt.ylabel('Price (USD)')
    plt.tight_layout()
    plt.savefig(os.path.join(SUB_DIRS['PRICING'], 'ram_pricing_tiers.png'), dpi=300)
    plt.close()

    # SSD Pricing Tiers
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=ssd_df, x='Size_GB', y='Price', hue='Protocol')
    plt.title('SSD Pricing Tiers: Capacity vs. Market Price', fontsize=14, fontweight='bold')
    plt.xlabel('Total Capacity (GB)')
    plt.ylabel('Price (USD)')
    plt.tight_layout()
    plt.savefig(os.path.join(SUB_DIRS['PRICING'], 'ssd_pricing_tiers.png'), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 4. Efficiency Analysis (Bar Charts)
    # ---------------------------------------------------------
    # PSU Efficiency Tax
    plt.figure(figsize=(10, 6))
    sns.barplot(data=psu_df, x='Efficiency Rating', y='Price', errorbar='sd', palette='viridis')
    plt.title('Average PSU Price by Efficiency Rating ("Efficiency Tax")', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.ylabel('Average Price (USD)')
    plt.tight_layout()
    plt.savefig(os.path.join(SUB_DIRS['EFFICIENCY'], 'psu_efficiency_tax.png'), dpi=300)
    plt.close()

    print(f"All EDA charts successfully saved to categorized folders in {OUTPUT_DIR}/")

# ---------------------------------------------------------
# 4. Feature 2: Predictive Machine Learning (Inference)
# ---------------------------------------------------------
def train_models():
    print("\nTraining Optimized Price Prediction Models...")
    models = {}

    gpu_features = ['Vram_GB', 'Boost_Clock_MHz', 'TDP_W']
    X_gpu = gpu_df[gpu_features]
    y_gpu = gpu_df['Price']
    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_gpu, y_gpu, test_size=0.2, random_state=42)
    g_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_g, y_train_g)
    models['gpu'] = {'model': g_model, 'features': gpu_features}
    print(f"GPU Model R2: {r2_score(y_test_g, g_model.predict(X_test_g)):.2f}")

    fi_gpu = pd.Series(g_model.feature_importances_, index=gpu_features).sort_values()
    plt.figure(figsize=(8, 5))
    fi_gpu.plot(kind='barh')
    plt.title('GPU Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'gpu_feature_importances.png'), dpi=300)
    plt.close()

    clean_cpu = cpu_df[(cpu_df['Price'] <= 1500) & (cpu_df['Cores'] <= 32)].copy()
    cpu_feat_raw = ['Cores', 'Threads', 'TDP_W', 'Base_Clock_GHz', 'Producer', 'Socket']
    X_c_raw = clean_cpu[cpu_feat_raw].dropna()
    y_c = clean_cpu.loc[X_c_raw.index, 'Price']
    X_c = pd.get_dummies(X_c_raw, columns=['Producer', 'Socket'], drop_first=True)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.2, random_state=42)
    c_model = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train_c, y_train_c)
    models['cpu'] = {'model': c_model, 'features': X_c.columns}
    print(f"CPU Model R2: {r2_score(y_test_c, c_model.predict(X_test_c)):.2f}")

    fi_cpu = pd.Series(c_model.feature_importances_, index=X_c.columns).sort_values()
    plt.figure(figsize=(10, 6))
    fi_cpu.plot(kind='barh')
    plt.title('CPU Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cpu_feature_importances.png'), dpi=300)
    plt.close()

    return models


def appraise_component(models):
    print("\n[ML Price Appraiser]")
    choice = input("Appraise [1] GPU or [2] CPU? ")

    if choice == '1':
        vram = float(input("VRAM (GB): "))
        boost = float(input("Boost Clock (MHz): "))
        tdp = float(input("TDP (W): "))
        sample = pd.DataFrame([[vram, boost, tdp]], columns=models['gpu']['features'])
        pred = models['gpu']['model'].predict(sample)[0]
        print(f"Predicted Fair Price: ${pred:.2f}")

    elif choice == '2':
        cores = int(input("Cores: "))
        threads = int(input("Threads: "))
        clock = float(input("Base Clock (GHz): "))
        tdp = float(input("TDP (W): "))
        prod = input("Producer (AMD/Intel): ")
        socket = input("Socket: ")

        sample = pd.DataFrame(0, index=[0], columns=models['cpu']['features'])
        sample.at[0, 'Cores'], sample.at[0, 'Threads'] = cores, threads
        sample.at[0, 'Base_Clock_GHz'], sample.at[0, 'TDP_W'] = clock, tdp
        prod_col = f'Producer_{prod}'
        socket_col = f'Socket_{socket}'
        if prod_col in sample.columns:
            sample.at[0, prod_col] = 1
        if socket_col in sample.columns:
            sample.at[0, socket_col] = 1

        pred = models['cpu']['model'].predict(sample)[0]
        print(f"Predicted Fair Price: ${pred:.2f}")
    else:
        print("Invalid selection. Returning to menu.")


def select_case(form_factor, max_price):
    candidates = case_df[(case_df['Price'] <= max_price) & case_df['Motherboard'].astype(str).str.contains(form_factor, na=False)]
    return candidates.sort_values(['Disk_3_5_Bays', 'Price'], ascending=[False, True]).head(1).iloc[0] if not candidates.empty else None


def select_ssd(max_price, min_size=500):
    candidates = ssd_df[(ssd_df['Price'] <= max_price) & (ssd_df['Size_GB'] >= min_size)].copy()
    if candidates.empty:
        return None
    candidates['Value'] = candidates['Size_GB'] / candidates['Price']
    return candidates.sort_values(['Value', 'Size_GB'], ascending=[False, False]).head(1).iloc[0]


def select_hdd(max_price):
    candidates = hdd_df[hdd_df['Price'] <= max_price].copy()
    if candidates.empty:
        return None
    return candidates.sort_values(['Size_GB', 'Price'], ascending=[False, True]).head(1).iloc[0]


def select_psu(min_watt, max_price):
    candidates = psu_df[(psu_df['Watt_W'] >= min_watt) & (psu_df['Price'] <= max_price)].copy()
    if candidates.empty:
        return None
    return candidates.sort_values(['Watt_W', 'Price'], ascending=[True, True]).head(1).iloc[0]


def recommend_pc(budget, use_case):
    use_case = use_case.lower()
    require_gpu = True
    min_gpu_vram = 0
    ram_min = 16
    psu_overhead = 1.2
    ssd_target = 500
    hdd_budget = 0

    if use_case == 'gaming':
        gpu_budget = budget * 0.45
        cpu_budget = budget * 0.25
    elif use_case == 'video editing':
        gpu_budget = budget * 0.25
        cpu_budget = budget * 0.35
        ram_min = 32
    elif use_case == 'home office':
        gpu_budget = 0
        cpu_budget = budget * 0.30
        require_gpu = False
        ram_min = 16
        ssd_target = 1000
    elif use_case == 'ai developer':
        gpu_budget = budget * 0.50
        cpu_budget = budget * 0.25
        ram_min = 32
        min_gpu_vram = 16
        psu_overhead = 1.3
        ssd_target = 1000
    elif use_case == 'home server':
        gpu_budget = 0
        cpu_budget = budget * 0.20
        require_gpu = False
        hdd_budget = budget * 0.4
    else:
        gpu_budget = budget * 0.30
        cpu_budget = budget * 0.30

    remaining = budget
    gpu = None
    if require_gpu:
        gpu_candidates = gpu_df[gpu_df['Price'] <= gpu_budget]
        gpu_candidates = gpu_candidates[gpu_candidates['Vram_GB'] >= min_gpu_vram] if min_gpu_vram > 0 else gpu_candidates
        if gpu_candidates.empty:
            print("No GPU fits the allocated budget or VRAM requirement.")
            return
        gpu = gpu_candidates.sort_values('Perf_Score', ascending=False).iloc[0]
        remaining -= gpu['Price']

    cpu_candidates = cpu_df[cpu_df['Price'] <= min(cpu_budget, remaining)].copy()
    if use_case == 'home office':
        cpu_candidates = cpu_candidates[cpu_candidates['Integrated_GPU'] == True]
    if cpu_candidates.empty:
        print("No CPU fits the allocated budget.")
        return
    cpu = cpu_candidates.sort_values(['Cores', 'Threads', 'Price'], ascending=[False, False, True]).iloc[0]
    remaining -= cpu['Price']

    mb_candidates = motherboard_df[(motherboard_df['Price'] <= remaining) & (motherboard_df['Socket'] == cpu['Socket'])].copy()
    if use_case == 'home server':
        mb_candidates = mb_candidates.sort_values(['SATA_Count', 'Price'], ascending=[False, True])
    else:
        mb_candidates = mb_candidates.sort_values('Price')
    if mb_candidates.empty:
        print("No compatible motherboard found.")
        return
    mb = mb_candidates.iloc[0]
    remaining -= mb['Price']

    ram_candidates = ram_df[(ram_df['Price'] <= remaining) & (ram_df['Size_GB'] >= ram_min) & ram_df['Ram_Type'].str.contains(mb['Memory Type'])].copy()
    if use_case == 'ai developer' and gpu is not None:
        ram_candidates = ram_candidates[ram_candidates['Size_GB'] >= max(ram_min, int(gpu['Vram_GB'] * 2))]
    if ram_candidates.empty:
        print("No compatible RAM found.")
        return
    ram = ram_candidates.sort_values(['Size_GB', 'Price'], ascending=[False, True]).iloc[0]
    remaining -= ram['Price']

    ssd = None
    if use_case != 'home server':
        ssd = select_ssd(remaining, min_size=ssd_target)
        if ssd is None and remaining > 0:
            ssd = select_ssd(remaining, min_size=0)
        if ssd is not None:
            remaining -= ssd['Price']

    total_tdp = cpu['TDP_W'] + (gpu['TDP_W'] if gpu is not None else 0)
    min_watt = total_tdp * psu_overhead
    psu = select_psu(min_watt, remaining)
    if psu is None:
        print("No compatible PSU found.")
        return
    remaining -= psu['Price']

    case = select_case(mb['Form Factor'], remaining)
    if case is None:
        print("No compatible case found.")
        return
    remaining -= case['Price']

    hdd = None
    if use_case == 'home server':
        hdd = select_hdd(hdd_budget)
        if hdd is None:
            print("No HDD found for home server budget allocation.")
            return
        remaining -= hdd['Price']

    total = cpu['Price'] + mb['Price'] + ram['Price'] + psu['Price'] + case['Price'] + (gpu['Price'] if gpu is not None else 0) + (ssd['Price'] if ssd is not None else 0) + (hdd['Price'] if hdd is not None else 0)
    print(f"\n=== PC Build Receipt ({use_case.title()}) ===")
    print(f"Budget: ${budget:.2f}")
    print(f"CPU: {cpu['Name']} - ${cpu['Price']:.2f}")
    print(f"Motherboard: {mb['Name']} - ${mb['Price']:.2f}")
    print(f"RAM: {ram['Name']} - ${ram['Price']:.2f}")
    if gpu is not None:
        print(f"GPU: {gpu['Name']} - ${gpu['Price']:.2f}")
    else:
        print("GPU: Integrated Graphics - $0.00")
    if ssd is not None:
        print(f"SSD: {ssd['Name']} - ${ssd['Price']:.2f}")
    print(f"PSU: {psu['Name']} - ${psu['Price']:.2f}")
    print(f"Case: {case['Name']} - ${case['Price']:.2f}")
    if hdd is not None:
        print(f"HDD: {hdd['Name']} - ${hdd['Price']:.2f}")
    print(f"\nTotal Cost: ${total:.2f}")
    print(f"Remaining Budget: ${remaining:.2f}")


if __name__ == "__main__":
    preprocess_data()
    models = train_models()

    while True:
        print("\nWelcome to the PC Hardware Tool")
        print("[1] Run Market Analysis (Generate EDA Charts)")
        print("[2] Build a PC (Recommendation Engine)")
        print("[3] Appraise a Component (ML Price Predictor)")
        print("[0] Exit")
        choice = input("Select an option: ")

        if choice == '1':
            generate_eda_charts()
        elif choice == '2':
            budget = float(input("Enter your total budget in USD: "))
            print("Select primary use case:")
            print("[1] Gaming")
            print("[2] Video Editing")
            print("[3] Home Office")
            print("[4] AI Developer")
            print("[5] Home Server")
            case_choice = input("Enter choice: ")
            choices = {
                '1': 'Gaming',
                '2': 'Video Editing',
                '3': 'Home Office',
                '4': 'AI Developer',
                '5': 'Home Server'
            }
            recommend_pc(budget, choices.get(case_choice, 'Gaming'))
        elif choice == '3':
            appraise_component(models)
        elif choice == '0':
            break
