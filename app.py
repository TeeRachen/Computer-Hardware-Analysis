import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

OUTPUT_DIR = 'outputs'
CORR_DIR = os.path.join(OUTPUT_DIR, 'Correlation_Matrices')
VALUE_DIR = os.path.join(OUTPUT_DIR, 'Value_Matrices')
PRICING_DIR = os.path.join(OUTPUT_DIR, 'Pricing_Tiers')
EFFICIENCY_DIR = os.path.join(OUTPUT_DIR, 'Efficiency_Analysis')
for output_dir in [OUTPUT_DIR, CORR_DIR, VALUE_DIR, PRICING_DIR, EFFICIENCY_DIR]:
    os.makedirs(output_dir, exist_ok=True)

# Load all datasets
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

print("All datasets loaded successfully!")
print(f"Case data shape: {case_df.shape}")
print(f"CPU Cooler data shape: {cpu_cooler_df.shape}")
print(f"CPU data shape: {cpu_df.shape}")
print(f"GPU data shape: {gpu_df.shape}")
print(f"HDD data shape: {hdd_df.shape}")
print(f"Monitor data shape: {monitor_df.shape}")
print(f"Motherboard data shape: {motherboard_df.shape}")
print(f"PSU data shape: {psu_df.shape}")
print(f"RAM data shape: {ram_df.shape}")
print(f"SSD data shape: {ssd_df.shape}")

# Data cleaning helpers

def clean_price(df):
    if 'Price' in df.columns:
        df['Price'] = df['Price'].astype(str)
        df['Price'] = df['Price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.replace(' USD', '', regex=False)
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    return df


def parse_size(size_str):
    if pd.isna(size_str):
        return None
    pieces = str(size_str).split()
    try:
        num = float(pieces[0])
    except ValueError:
        return None
    if len(pieces) > 1 and 'TB' in pieces[1].upper():
        return num * 1000
    return num


def extract_numeric(series, pattern):
    return series.astype(str).str.extract(pattern).astype(float)


def preprocess_data():
    global gpu_df, cpu_df, monitor_df, psu_df, ram_df, ssd_df, hdd_df, motherboard_df, case_df

    gpu_df = clean_price(gpu_df)
    cpu_df = clean_price(cpu_df)
    monitor_df = clean_price(monitor_df)
    psu_df = clean_price(psu_df)
    ram_df = clean_price(ram_df)
    ssd_df = clean_price(ssd_df)
    hdd_df = clean_price(hdd_df)
    motherboard_df = clean_price(motherboard_df)
    case_df = clean_price(case_df)

    gpu_df['Vram_GB'] = extract_numeric(gpu_df['Vram'], r'(\d+)')
    gpu_df['Boost_Clock_MHz'] = extract_numeric(gpu_df['Boost Clock'], r'(\d+)')
    gpu_df['TDP_W'] = extract_numeric(gpu_df['TDP'], r'(\d+)')
    gpu_df['Perf_Score'] = (gpu_df['Boost_Clock_MHz'] * gpu_df['Vram_GB']) / 1000

    cpu_df['Cores'] = pd.to_numeric(cpu_df['Cores'], errors='coerce')
    cpu_df['Threads'] = pd.to_numeric(cpu_df['Threads'], errors='coerce')
    cpu_df['TDP_W'] = extract_numeric(cpu_df['TDP'], r'(\d+)')
    cpu_df['Base_Clock_GHz'] = extract_numeric(cpu_df['Base Clock'], r'(\d+\.?\d*)')

    monitor_df['Refresh_Rate_Hz'] = extract_numeric(monitor_df['Refresh Rate'], r'(\d+)')
    psu_df['Watt_W'] = extract_numeric(psu_df['Watt'], r'(\d+)')

    ram_df['Size_GB'] = extract_numeric(ram_df['Size'], r'(\d+)')
    ram_df['Clock_MHz'] = pd.to_numeric(ram_df['Clock'], errors='coerce')
    ram_df['Ram_Type'] = ram_df['Ram Type'].astype(str).str.extract(r'(DDR\d+)')
    ram_df['Size_Category'] = ram_df['Size_GB'].apply(lambda x: f"{int(x)}GB" if pd.notna(x) else 'Unknown')

    ssd_df['Size_GB'] = ssd_df['Size'].apply(parse_size)
    ssd_df['Protocol'] = ssd_df['Protocol'].replace({'NVM': 'NVMe'}).fillna('Unknown')
    ssd_df['Form_Factor'] = ssd_df['Form Factor'].fillna('Unknown')

    hdd_df['Size_GB'] = hdd_df['Size'].apply(parse_size)
    hdd_df['Form_Factor'] = hdd_df['Form Factor'].fillna('Unknown')

    motherboard_df['SATA_Count'] = extract_numeric(motherboard_df['SATA'].astype(str), r'(\d+)')
    cpu_df['Integrated_GPU'] = cpu_df['Integrated GPU'].astype(str).str.lower().isin(['true', 'yes']) | cpu_df['Integrated GPU'].astype(str).str.contains(r'(radeon|vega|uhd|iris|graphics|intel)', case=False)
    disk_col = find_column(case_df, 'Disk 3.5')
    if disk_col is not None:
        case_df['Disk_3_5_Bays'] = extract_numeric(case_df[disk_col].astype(str), r'(\d+)')
    else:
        case_df['Disk_3_5_Bays'] = None

    gpu_df.dropna(subset=['Price', 'Vram_GB', 'Boost_Clock_MHz'], inplace=True)
    cpu_df.dropna(subset=['Price', 'Cores', 'Base_Clock_GHz'], inplace=True)
    monitor_df.dropna(subset=['Price', 'Refresh_Rate_Hz'], inplace=True)
    psu_df.dropna(subset=['Price', 'Watt_W'], inplace=True)
    ram_df.dropna(subset=['Price', 'Size_GB', 'Ram_Type'], inplace=True)
    ssd_df.dropna(subset=['Price', 'Size_GB', 'Protocol'], inplace=True)
    hdd_df.dropna(subset=['Price', 'Size_GB'], inplace=True)
    motherboard_df.dropna(subset=['Price', 'Socket'], inplace=True)
    case_df.dropna(subset=['Price', 'Motherboard'], inplace=True)


def output_path(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def find_column(df, substring):
    for col in df.columns:
        if substring in col:
            return col
    return None


def plot_scatter(df, x_col, y_col, title, filename, color_col=None, alpha=0.4):
    if x_col not in df.columns or y_col not in df.columns:
        print(f"Skipping {title}: Missing columns")
        return
    df_plot = df.dropna(subset=[x_col, y_col])
    if df_plot.empty:
        print(f"Skipping {title}: No data")
        return
    plt.figure(figsize=(10, 6))
    if color_col and color_col in df.columns:
        categories = df_plot[color_col].dropna().unique()
        for cat in categories:
            subset = df_plot[df_plot[color_col] == cat]
            plt.scatter(subset[x_col], subset[y_col], alpha=alpha, label=str(cat))
        plt.legend(title=color_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(df_plot[x_col], df_plot[y_col], alpha=alpha)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(output_path(filename), dpi=300)
    plt.close()
    print(f"Saved {output_path(filename)}")


def plot_boxplot(df, x_col, y_col, title, filename, hue_col=None):
    if x_col not in df.columns or y_col not in df.columns:
        print(f"Skipping {title}: Missing columns")
        return
    df_plot = df.dropna(subset=[x_col, y_col])
    if df_plot.empty:
        print(f"Skipping {title}: No data")
        return
    groups = df_plot.groupby(x_col)[y_col]
    labels = []
    values = []
    for name, group in groups:
        labels.append(str(name))
        values.append(group.values)
    if not values:
        print(f"Skipping {title}: No groups")
        return
    plt.figure(figsize=(12, 6))
    plt.boxplot(values, tick_labels=labels, patch_artist=True)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path(filename), dpi=300)
    plt.close()
    print(f"Saved {output_path(filename)}")


def plot_grouped_line(df, x_col, y_col, group_col, title, filename):
    if x_col not in df.columns or y_col not in df.columns or group_col not in df.columns:
        print(f"Skipping {title}: Missing columns")
        return
    df_plot = df.dropna(subset=[x_col, y_col, group_col])
    if df_plot.empty:
        print(f"Skipping {title}: No data")
        return
    plt.figure(figsize=(10, 6))
    for group, subset in df_plot.groupby(group_col):
        bucketed = subset.groupby(x_col)[y_col].mean().sort_index()
        plt.plot(bucketed.index, bucketed.values, marker='o', label=str(group))
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(title=group_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path(filename), dpi=300)
    plt.close()
    print(f"Saved {output_path(filename)}")


def plot_ram_price_capacity_by_type(df, title, filename):
    if 'Size_Category' not in df.columns or 'Price' not in df.columns or 'Ram_Type' not in df.columns:
        print(f"Skipping {title}: Missing columns")
        return
    df_plot = df.dropna(subset=['Size_Category', 'Price', 'Ram_Type'])
    if df_plot.empty:
        print(f"Skipping {title}: No data")
        return
    groups = []
    labels = []
    colors = []
    type_palette = {'DDR4': '#4c72b0', 'DDR5': '#dd8452'}
    for ram_type in sorted(df_plot['Ram_Type'].unique()):
        sizes = sorted(df_plot[df_plot['Ram_Type'] == ram_type]['Size_Category'].unique(), key=lambda x: int(x.replace('GB', '')) if isinstance(x, str) and x.endswith('GB') else x)
        for size in sizes:
            group = df_plot[(df_plot['Ram_Type'] == ram_type) & (df_plot['Size_Category'] == size)]['Price']
            if not group.empty:
                groups.append(group.values)
                labels.append(f"{ram_type} {size}")
                colors.append(type_palette.get(ram_type, '#7f7f7f'))
    if not groups:
        print(f"Skipping {title}: No grouped data")
        return
    plt.figure(figsize=(14, 7))
    box = plt.boxplot(groups, labels=labels, patch_artist=True)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.title(title)
    plt.xlabel('RAM Type and Capacity')
    plt.ylabel('Price')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path(filename), dpi=300)
    plt.close()
    print(f"Saved {output_path(filename)}")


def plot_storage_protocol_boxplot(df, title, filename):
    if 'Size_GB' not in df.columns or 'Price' not in df.columns or 'Protocol' not in df.columns:
        print(f"Skipping {title}: Missing columns")
        return
    df_plot = df.dropna(subset=['Size_GB', 'Price', 'Protocol'])
    if df_plot.empty:
        print(f"Skipping {title}: No data")
        return
    groups = []
    labels = []
    colors = []
    palette = {'NVMe': '#4c72b0', 'SATA': '#dd8452'}
    for proto in sorted(df_plot['Protocol'].unique()):
        sizes = sorted(df_plot[df_plot['Protocol'] == proto]['Size_GB'].unique())
        for size in sizes:
            group = df_plot[(df_plot['Protocol'] == proto) & (df_plot['Size_GB'] == size)]['Price']
            if not group.empty:
                groups.append(group.values)
                labels.append(f"{proto} {int(size)}GB")
                colors.append(palette.get(proto, '#7f7f7f'))
    if not groups:
        print(f"Skipping {title}: No grouped data")
        return
    plt.figure(figsize=(14, 7))
    box = plt.boxplot(groups, labels=labels, patch_artist=True)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.title(title)
    plt.xlabel('Storage Protocol and Capacity')
    plt.ylabel('Price')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path(filename), dpi=300)
    plt.close()
    print(f"Saved {output_path(filename)}")


def plot_price_per_gb(df, size_col, price_col, group_col, title, filename):
    if size_col not in df.columns or price_col not in df.columns or group_col not in df.columns:
        print(f"Skipping {title}: Missing columns")
        return
    df_plot = df.dropna(subset=[size_col, price_col, group_col])
    df_plot = df_plot[df_plot[size_col] > 0]
    if df_plot.empty:
        print(f"Skipping {title}: No data")
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
    plt.savefig(output_path(filename), dpi=300)
    plt.close()
    print(f"Saved {output_path(filename)}")


def plot_psu_efficiency(df):
    df_plot = df.dropna(subset=['Efficiency Rating', 'Price'])
    if df_plot.empty:
        print("Skipping PSU efficiency chart: no data")
        return
    stats = df_plot.groupby('Efficiency Rating')['Price'].agg(['mean', 'std', 'count']).reset_index()
    stats = stats[stats['count'] >= 1]
    plt.figure(figsize=(10, 6))
    plt.bar(stats['Efficiency Rating'], stats['mean'], yerr=stats['std'].fillna(0), alpha=0.8)
    plt.title('Average PSU Price by Efficiency Rating')
    plt.xlabel('Efficiency Rating')
    plt.ylabel('Average Price (USD)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path(os.path.join('Efficiency_Analysis', 'psu_efficiency_price.png')), dpi=300)
    plt.close()
    print(f"Saved {output_path(os.path.join('Efficiency_Analysis', 'psu_efficiency_price.png'))}")


def plot_correlation_heatmap(df, columns, title, filename):
    df_plot = df[columns].dropna()
    if df_plot.empty:
        print(f"Skipping {title}: no numeric data")
        return
    corr = df_plot.corr()
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    ticks = range(len(columns))
    plt.xticks(ticks, columns, rotation=45, ha='right')
    plt.yticks(ticks, columns)
    for i in range(len(columns)):
        for j in range(len(columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', color='black')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path(os.path.join('Correlation_Matrices', filename)), dpi=300)
    plt.close()
    print(f"Saved {output_path(os.path.join('Correlation_Matrices', filename))}")


def generate_eda_charts():
    print("\nGenerating EDA charts...")
    plot_correlation_heatmap(gpu_df, ['Price', 'Vram_GB', 'Boost_Clock_MHz', 'TDP_W', 'Perf_Score'], 'GPU Feature Correlation', 'gpu_correlation_heatmap.png')
    plot_correlation_heatmap(cpu_df, ['Price', 'Cores', 'Threads', 'TDP_W', 'Base_Clock_GHz'], 'CPU Feature Correlation', 'cpu_correlation_heatmap.png')

    plot_scatter(gpu_df, 'Price', 'Perf_Score', 'GPU: Price vs Performance Score', os.path.join('Value_Matrices', 'gpu_value_matrix.png'), 'Producer')
    plot_scatter(cpu_df, 'Price', 'Cores', 'CPU: Price vs Cores', os.path.join('Value_Matrices', 'cpu_value_matrix.png'), 'Producer')
    plot_scatter(monitor_df, 'Price', 'Refresh_Rate_Hz', 'Monitor: Price vs Refresh Rate', os.path.join('Value_Matrices', 'monitor_value_matrix.png'), 'Resolution')

    plot_scatter(ram_df, 'Size_GB', 'Price', 'RAM: Price vs Capacity', os.path.join('Pricing_Tiers', 'ram_capacity_price.png'), 'Ram_Type')
    plot_scatter(ram_df, 'Clock_MHz', 'Price', 'RAM: Price vs Clock Speed', os.path.join('Pricing_Tiers', 'ram_clock_price.png'), 'Ram_Type')
    plot_ram_price_capacity_by_type(ram_df, 'RAM Price Distribution by Capacity and Type', os.path.join('Pricing_Tiers', 'ram_capacity_by_type_boxplot.png'))

    plot_storage_protocol_boxplot(ssd_df, 'SSD Price by Size and Protocol', os.path.join('Pricing_Tiers', 'ssd_protocol_size_boxplot.png'))
    plot_price_per_gb(ssd_df, 'Size_GB', 'Price', 'Protocol', 'SSD Price per GB by Size and Protocol', os.path.join('Pricing_Tiers', 'ssd_price_per_gb_line.png'))
    plot_scatter(hdd_df, 'Size_GB', 'Price', 'HDD: Price vs Size', os.path.join('Pricing_Tiers', 'hdd_size_price.png'), 'Form_Factor')
    plot_grouped_line(hdd_df, 'Size_GB', 'Price', 'Form_Factor', 'HDD Mean Price by Size and Form Factor', os.path.join('Pricing_Tiers', 'hdd_price_per_gb.png'))

    plot_psu_efficiency(psu_df)
    print("EDA chart generation complete.")


def train_models():
    print("\nTraining price prediction models...")
    models = {}

    gpu_features = ['Vram_GB', 'Boost_Clock_MHz', 'TDP_W']
    X_gpu = gpu_df[gpu_features]
    y_gpu = gpu_df['Price']
    X_train_gpu, X_test_gpu, y_train_gpu, y_test_gpu = train_test_split(X_gpu, y_gpu, test_size=0.2, random_state=42)
    gpu_model = RandomForestRegressor(random_state=42)
    gpu_model.fit(X_train_gpu, y_train_gpu)
    y_pred_gpu = gpu_model.predict(X_test_gpu)
    mae_gpu = mean_absolute_error(y_test_gpu, y_pred_gpu)
    r2_gpu = r2_score(y_test_gpu, y_pred_gpu)
    print(f"GPU Price Prediction - MAE: ${mae_gpu:.2f}, R²: {r2_gpu:.2f}")

    plt.figure(figsize=(8, 5))
    importances = pd.Series(gpu_model.feature_importances_, index=gpu_features).sort_values()
    importances.plot(kind='barh')
    plt.title('GPU Feature Importances for Price Prediction')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(output_path('gpu_feature_importances.png'), dpi=300)
    plt.close()
    print(f"Saved {output_path('gpu_feature_importances.png')}")
    models['gpu'] = gpu_model

    cpu_features = ['Cores', 'Threads', 'TDP_W', 'Base_Clock_GHz']
    X_cpu = cpu_df[cpu_features].dropna()
    y_cpu = cpu_df.loc[X_cpu.index, 'Price']
    X_train_cpu, X_test_cpu, y_train_cpu, y_test_cpu = train_test_split(X_cpu, y_cpu, test_size=0.2, random_state=42)
    cpu_model = RandomForestRegressor(random_state=42)
    cpu_model.fit(X_train_cpu, y_train_cpu)
    y_pred_cpu = cpu_model.predict(X_test_cpu)
    mae_cpu = mean_absolute_error(y_test_cpu, y_pred_cpu)
    r2_cpu = r2_score(y_test_cpu, y_pred_cpu)
    print(f"CPU Price Prediction - MAE: ${mae_cpu:.2f}, R²: {r2_cpu:.2f}")

    plt.figure(figsize=(8, 5))
    importances = pd.Series(cpu_model.feature_importances_, index=cpu_features).sort_values()
    importances.plot(kind='barh')
    plt.title('CPU Feature Importances for Price Prediction')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(output_path('cpu_feature_importances.png'), dpi=300)
    plt.close()
    print(f"Saved {output_path('cpu_feature_importances.png')}")
    models['cpu'] = cpu_model

    return models


def appraise_gpu(model):
    print("\nGPU Price Appraiser")
    vram = float(input("Enter the VRAM in GB: "))
    boost = float(input("Enter the Boost Clock in MHz: "))
    tdp = float(input("Enter the TDP in W: "))
    sample = pd.DataFrame([{'Vram_GB': vram, 'Boost_Clock_MHz': boost, 'TDP_W': tdp}])
    price = model.predict(sample)[0]
    print(f"Predicted fair market price: ${price:.2f}")


def appraise_cpu(model):
    print("\nCPU Price Appraiser")
    cores = int(input("Enter the number of cores: "))
    threads = int(input("Enter the number of threads: "))
    base_clock = float(input("Enter the Base Clock in GHz: "))
    tdp = float(input("Enter the TDP in W: "))
    sample = pd.DataFrame([{'Cores': cores, 'Threads': threads, 'TDP_W': tdp, 'Base_Clock_GHz': base_clock}])
    price = model.predict(sample)[0]
    print(f"Predicted fair market price: ${price:.2f}")


def appraise_component(models):
    while True:
        print("\nAppraise a component:")
        print("[1] GPU")
        print("[2] CPU")
        print("[0] Back")
        choice = input("Select an option: ")
        if choice == '1':
            appraise_gpu(models['gpu'])
            break
        elif choice == '2':
            appraise_cpu(models['cpu'])
            break
        elif choice == '0':
            break
        else:
            print("Invalid selection. Choose 1, 2, or 0.")


def select_case(form_factor, max_price):
    case = case_df[(case_df['Price'] <= max_price) & case_df['Motherboard'].astype(str).str.contains(form_factor, na=False)].sort_values(['Disk_3_5_Bays', 'Price'], ascending=[False, True])
    return case.iloc[0] if not case.empty else None


def select_ssd(max_price, min_size=0, prefer_size=None):
    candidates = ssd_df[(ssd_df['Price'] <= max_price) & (ssd_df['Size_GB'] >= min_size)].copy()
    if candidates.empty:
        return None
    if prefer_size is not None:
        larger = candidates[candidates['Size_GB'] >= prefer_size]
        if not larger.empty:
            candidates = larger
    candidates['Value'] = candidates['Size_GB'] / candidates['Price']
    return candidates.sort_values(['Value', 'Size_GB'], ascending=[False, False]).iloc[0]


def select_hdd(max_price):
    candidates = hdd_df[hdd_df['Price'] <= max_price].copy()
    if candidates.empty:
        return None
    return candidates.sort_values(['Size_GB', 'Price'], ascending=[False, True]).iloc[0]


def recommend_pc(budget, use_case):
    if use_case.lower() == 'gaming':
        gpu_budget = budget * 0.45
        cpu_budget = budget * 0.25
        ram_min_size = 16
        psu_overhead = 1.2
        require_gpu = True
        min_gpu_vram = 0
        target_ssd_size = 500
    elif use_case.lower() in ('productivity', 'video editing'):
        gpu_budget = budget * 0.25
        cpu_budget = budget * 0.35
        ram_min_size = 32
        psu_overhead = 1.2
        require_gpu = True
        min_gpu_vram = 0
        target_ssd_size = 500
    elif use_case.lower() == 'home office':
        gpu_budget = 0
        cpu_budget = budget * 0.3
        ram_min_size = 16
        psu_overhead = 1.2
        require_gpu = False
        min_gpu_vram = 0
        target_ssd_size = 1000
    elif use_case.lower() == 'ai developer':
        gpu_budget = budget * 0.5
        cpu_budget = budget * 0.25
        ram_min_size = 32
        psu_overhead = 1.3
        require_gpu = True
        min_gpu_vram = 16
        target_ssd_size = 1000
    elif use_case.lower() == 'home server':
        gpu_budget = 0
        cpu_budget = budget * 0.2
        ram_min_size = 16
        psu_overhead = 1.2
        require_gpu = False
        min_gpu_vram = 0
        target_ssd_size = 500
    else:
        gpu_budget = budget * 0.3
        cpu_budget = budget * 0.3
        ram_min_size = 16
        psu_overhead = 1.2
        require_gpu = True
        min_gpu_vram = 0
        target_ssd_size = 500

    remaining = budget
    gpu = None
    if require_gpu:
        gpu_candidates = gpu_df[gpu_df['Price'] <= gpu_budget]
        if min_gpu_vram > 0:
            gpu_candidates = gpu_candidates[gpu_candidates['Vram_GB'] >= min_gpu_vram]
        gpu = gpu_candidates.sort_values('Perf_Score', ascending=False).iloc[0] if not gpu_candidates.empty else None
        if gpu is None:
            print("No GPU fits allocated budget or VRAM requirement")
            return
        remaining -= gpu['Price']
    else:
        gpu = pd.Series({'TDP_W': 0.0, 'Vram_GB': 0.0, 'Name': 'Integrated Graphics', 'Price': 0.0})

    cpu_candidates = cpu_df[cpu_df['Price'] <= min(cpu_budget, remaining)].copy()
    if use_case.lower() == 'home office':
        cpu_candidates = cpu_candidates[cpu_candidates['Integrated_GPU'] == True]
    if use_case.lower() == 'home server':
        cpu_candidates = cpu_candidates.sort_values(['TDP_W', 'Price'], ascending=[True, True])
    elif use_case.lower() == 'ai developer':
        cpu_candidates = cpu_candidates.sort_values(['Cores', 'Threads', 'Price'], ascending=[False, False, True])
    else:
        cpu_candidates = cpu_candidates.sort_values(['Cores', 'Threads', 'Price'], ascending=[False, False, True])
    cpu = cpu_candidates.iloc[0] if not cpu_candidates.empty else None
    if cpu is None:
        print("No CPU fits allocated budget")
        return
    remaining -= cpu['Price']
    socket = cpu['Socket']

    mb_candidates = motherboard_df[(motherboard_df['Price'] <= remaining) & (motherboard_df['Socket'] == socket)].copy()
    if use_case.lower() == 'home server':
        mb_candidates = mb_candidates.sort_values(['SATA_Count', 'Price'], ascending=[False, True])
    else:
        mb_candidates = mb_candidates.sort_values('Price')
    mb = mb_candidates.iloc[0] if not mb_candidates.empty else None
    if mb is None:
        print("No compatible Motherboard")
        return
    remaining -= mb['Price']
    mem_type = mb['Memory Type']

    ram_candidates = ram_df[(ram_df['Price'] <= remaining) & (ram_df['Ram Type'].str.startswith(mem_type)) & (ram_df['Size_GB'] >= ram_min_size)].copy()
    if use_case.lower() == 'ai developer':
        ram_candidates = ram_candidates[ram_candidates['Size_GB'] >= max(ram_min_size, int(gpu['Vram_GB'] * 2))]
    ram = ram_candidates.sort_values(['Size_GB', 'Price'], ascending=[False, True]).iloc[0] if not ram_candidates.empty else None
    if ram is None:
        print("No compatible RAM")
        return
    remaining -= ram['Price']

    ssd = select_ssd(remaining, min_size=target_ssd_size)
    if ssd is None and target_ssd_size > 0:
        ssd = select_ssd(remaining, min_size=0)
    if ssd is not None:
        remaining -= ssd['Price']

    total_tdp = cpu['TDP_W'] + gpu['TDP_W']
    min_watt = total_tdp * psu_overhead
    psu_candidates = psu_df[(psu_df['Price'] <= remaining) & (psu_df['Watt_W'] >= min_watt)].sort_values('Price')
    psu = psu_candidates.iloc[0] if not psu_candidates.empty else None
    if psu is None:
        print("No compatible PSU")
        return
    remaining -= psu['Price']

    case = select_case(mb['Form Factor'], remaining)
    if case is None:
        print("No compatible Case")
        return
    remaining -= case['Price']

    hdd = None
    if use_case.lower() == 'home server':
        hdd = select_hdd(budget * 0.4)
        if hdd is None:
            print("No HDD fits home server budget allocation")
            return

    total = cpu['Price'] + mb['Price'] + ram['Price'] + psu['Price'] + case['Price'] + (gpu['Price'] if require_gpu else 0.0) + (ssd['Price'] if ssd is not None else 0.0) + (hdd['Price'] if hdd is not None else 0.0)
    remaining_budget = budget - total

    print("\n=== PC Build Receipt ===")
    print(f"Budget: ${budget}")
    print(f"Use Case: {use_case}")
    print("\nComponents:")
    print(f"CPU: {cpu['Name']} - ${cpu['Price']:.2f}")
    print(f"Motherboard: {mb['Name']} - ${mb['Price']:.2f}")
    print(f"RAM: {ram['Name']} - ${ram['Price']:.2f}")
    if ssd is not None:
        print(f"SSD: {ssd['Name']} - ${ssd['Price']:.2f}")
    if require_gpu:
        print(f"GPU: {gpu['Name']} - ${gpu['Price']:.2f}")
    else:
        print("GPU: Integrated Graphics - $0.00")
    print(f"PSU: {psu['Name']} - ${psu['Price']:.2f}")
    print(f"Case: {case['Name']} - ${case['Price']:.2f}")
    if hdd is not None:
        print(f"HDD: {hdd['Name']} - ${hdd['Price']:.2f}")
    print(f"\nTotal Cost: ${total:.2f}")
    print(f"Remaining Budget: ${remaining_budget:.2f}")


def input_float(prompt):
    while True:
        try:
            value = float(input(prompt))
            if value <= 0:
                print("Value must be positive.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a numeric value.")


def menu_build_pc():
    budget = input_float("Enter your total budget in USD: ")
    while True:
        print("Select primary use case:")
        print("[1] Gaming")
        print("[2] Video Editing")
        print("[3] Home Office")
        print("[4] AI Developer")
        print("[5] Home Server")
        choice = input("Enter choice: ")
        if choice == '1':
            use_case = 'Gaming'
            break
        elif choice == '2':
            use_case = 'Video Editing'
            break
        elif choice == '3':
            use_case = 'Home Office'
            break
        elif choice == '4':
            use_case = 'AI Developer'
            break
        elif choice == '5':
            use_case = 'Home Server'
            break
        else:
            print("Invalid selection. Please choose 1-5.")
    recommend_pc(budget, use_case)


def main_menu(models):
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
            menu_build_pc()
        elif choice == '3':
            appraise_component(models)
        elif choice == '0':
            print("Exiting application.")
            break
        else:
            print("Invalid selection. Please choose 0-3.")


if __name__ == "__main__":
    preprocess_data()
    trained_models = train_models()
    main_menu(trained_models)
