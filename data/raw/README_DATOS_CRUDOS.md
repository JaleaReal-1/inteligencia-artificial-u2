# Microdatos Crudos ENAHO y ESCALE 2025

Los archivos crudos CSV (enaho_m100_2025.csv, enaho_m200_2025.csv, enaho_m300_2025.csv, enaho_sumaria_2025.csv, escale_padron.csv, escale_padlocal.csv) superan los 900 MB en conjunto (Módulo 300 = 556 MB).
Por políticas de GitHub (límite de 100 MB por archivo), estos archivos no se suben al repositorio remoto.

El dataset procesado y limpio ya se encuentra disponible y listo para usar en:
data/processed/dataset_enaho_escale_2025.parquet (1.1 MB) y dataset_enaho_escale_2025.csv (7.1 MB).

Si desea regenerar el pipeline desde cero a partir de los datos crudos del INEI/MINEDU:
1. Descargue los ZIPs del repositorio oficial compartido por la cátedra o los portales de INEI y ESCALE.
2. Coloque los 6 archivos CSV filtrados en esta carpeta data/raw/.
3. Ejecute el script src/feature_engineering.py o el notebook 
otebooks/01_ingenieria_datos.ipynb.
