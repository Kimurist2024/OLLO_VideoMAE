import os

# AVIONディレクトリ内のファイルを確認
avion_data_path = "/home/ollo/VideoMAE/AVION/avion/data"
if os.path.exists(avion_data_path):
    print("Available files in AVION/avion/data:")
    for file in os.listdir(avion_data_path):
        print(f"  - {file}")