from data_utils import full_clean_pipeline

raw_csv_path = r"C:\Users\essaf\Desktop\MLopsTP\Mlpro\dataSet\car_data.csv"
output_csv_path = r'C:\Users\essaf\Desktop\MLopsTP\Mlpro\dataSet\cleaned_cardata3.csv'

cleaned_data = full_clean_pipeline(raw_csv_path, output_csv_path)

print(f"\n✅ Nettoyage terminé!")
print(f"Shape finale: {cleaned_data.shape}")
print(f"\nAperçu des données:")
print(cleaned_data.head())