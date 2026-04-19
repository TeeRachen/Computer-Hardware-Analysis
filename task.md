## Task
Create an data analysis program (app.py) where it imported the datasets and implemented the following features. The results should be exported as image file.

## Datasets
Folder: Dataset
- CaseData.csv
- CPUCoolerData.csv
- CPUData.csv
- GPUData.csv
- HDDData.csv
- MonitorData.csv
- MotherboardData.csv
- PSUData.csv
- RAMData.csv
- SSDData.csv

## Features
- EDA. Value for Money (Performance-per-Dollar Matrix): Create scatter plots to find the "sweet spot" for buyers.
    Example
    GPU: Price vs. VRAM or Price vs. Boost Clock.
    CPU: Price vs. Core/Thread count.
    Monitor: Price vs. Refresh Rate / Resolution.
- Recommendation System. The "PC Part Picker" A user inputs their budget (e.g., $1000) and primary use case (e.g., 1440p Gaming, Video Editing). The script filters through the datasets and suggests a CPU, GPU, Motherboard, RAM, PSU, and Case that fit within the budget and are mutually compatible (matching Socket, Form Factor, Wattage vs TDP).