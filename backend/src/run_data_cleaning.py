from data_utils import full_clean_pipeline

raw_csv_path = r"Mlpro\dataSet\car_data.csv"
output_csv_path = r'Mlpro\dataSet\cleaned_cardata3.csv'

cleaned_data = full_clean_pipeline(raw_csv_path, output_csv_path)

print(f"\n✅ Nettoyage terminé!")
print(f"Shape finale: {cleaned_data.shape}")
print(f"\nAperçu des données:")
print(cleaned_data.head())