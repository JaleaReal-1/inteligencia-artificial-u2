"""
Configuración global del proyecto.
PROYECTO IA: BRECHA DIGITAL Y RENDIMIENTO ESCOLAR EN EL PERÚ (CORTE TRANSVERSAL 2025)
Escuela Profesional de Ingeniería de Sistemas - Curso de Inteligencia Artificial
"""
from pathlib import Path

# Semilla global para reproducibilidad total
RANDOM_STATE = 42

# Rutas base del proyecto
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DOCS_DIR = DATA_DIR / "docs"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Archivos de microdatos crudos (corte anual 2025 filtrado de ENAHO Panel 1041 y ESCALE)
ENAHO_M100 = RAW_DIR / "enaho_m100_2025.csv"
ENAHO_M200 = RAW_DIR / "enaho_m200_2025.csv"
ENAHO_M300 = RAW_DIR / "enaho_m300_2025.csv"
ENAHO_SUMARIA = RAW_DIR / "enaho_sumaria_2025.csv"
ESCALE_PADRON = RAW_DIR / "escale_padron.csv"
ESCALE_PADLOCAL = RAW_DIR / "escale_padlocal.csv"

# Datasets procesados de salida
DATASET_FINAL_PARQUET = PROCESSED_DIR / "dataset_enaho_escale_2025.parquet"
DATASET_FINAL_CSV = PROCESSED_DIR / "dataset_enaho_escale_2025.csv"
DICCIONARIO_VARIABLES = PROCESSED_DIR / "diccionario_variables.csv"

# Claves primarias relacionales para el cruce (corte 2025)
KEYS_HOGAR = ["conglome_25", "vivienda_25", "hogar_25"]
KEYS_PERSONA = KEYS_HOGAR + ["codperso_25"]
KEY_UBIGEO_ENAHO = "ubigeo_25"
KEY_UBIGEO_ESCALE = "codgeo"

# Columnas que DEBEN preservarse como STRING con ceros a la izquierda (zfill)
STR_KEYS = [
    "numpanh", "conglome", "vivienda", "conglome_25", "vivienda_25",
    "hogar_25", "codperso_25", "ubigeo_25", "codgeo", "codlocal", "cod_mod"
]

# Población objetivo: Estudiantes en edad escolar obligatoria
EDAD_MIN = 6
EDAD_MAX = 17

# ==============================================================================
# VARIABLES EDUCATIVAS Y DEMOGRÁFICAS (Módulos 200 y 300 - INEI)
# ==============================================================================
COL_SEXO = "p207_25"            # 1=Hombre, 2=Mujer
COL_EDAD = "p208a_25"           # Edad en años cumplidos (6 a 17)
COL_PARENTESCO = "p203_25"      # Relación con el jefe del hogar

COL_LENGUA = "p300a_25"         # Lengua materna aprendida en la niñez (1=Castellano, 2=Quechua, 3=Aimara...)
COL_MATRICULA = "p306_25"       # ¿Está matriculado en el año escolar actual? (1=Sí, 2=No)
COL_ASISTE = "p307_25"          # ¿Asiste actualmente a la escuela? (1=Sí, 2=No)
COL_NIVEL_ACTUAL = "p308a_25"   # Nivel educativo que cursa en el año actual (1=Inicial, 2=Primaria, 3=Secundaria...)
COL_GRADO_ACTUAL = "p308b_25"   # Grado o año de estudios actual (1 a 6 en Primaria, 1 a 5 en Secundaria)
COL_TIPO_GESTION = "p301d_25"   # Gestión de la IE donde asiste (1=Estatal, 2=No estatal)

# ==============================================================================
# VARIABLES DE TIC Y CONECTIVIDAD (Módulos 100 y 300)
# ==============================================================================
COL_INTERNET_HOGAR = "p1144_25" # Conexión a Internet en el hogar (1=Sí, 0/2=No)
COL_LUZ_ELECTRICA = "p1121_25"  # Alumbrado eléctrico por red pública (1=Sí, 0/2=No)
COL_PC_HOGAR = "p1141_25"       # Computadora en el hogar (1=Sí, 0/2=No)
COL_LAPTOP_HOGAR = "p1142_25"   # Laptop en el hogar (1=Sí, 0/2=No)
COL_TABLET_HOGAR = "p1143_25"   # Tablet en el hogar (1=Sí, 0/2=No)
COL_USO_INTERNET_INDIV = "p314a_25"  # Uso de internet en el mes anterior (1=Sí, 2=No)
COL_LUGAR_INTERNET = "p314b_1_25"    # Uso de internet en el hogar (mes anterior)

# ==============================================================================
# VARIABLES SOCIOECONÓMICAS Y FACTOR DE EXPANSIÓN (Sumaria)
# ==============================================================================
COL_GASTO_PC = "gashog2d_25"    # Gasto per cápita mensual neto del hogar
COL_INGRESO_PC = "inghog2d_25"  # Ingreso per cápita mensual neto del hogar
COL_POBREZA = "pobreza_25"      # Condición de pobreza (1=Extremo, 2=No extremo, 3=No pobre)
COL_ESTRATO = "estrato_25"      # Estrato geográfico (1 a 8, urbano a rural)
COL_DOMINIO = "dominio_25"      # Dominio geográfico (Costa, Sierra, Selva, Lima)
COL_FACTOR = "factor07_25"      # Factor de expansión poblacional anual

# ==============================================================================
# VARIABLE DE AGRUPACIÓN PARA PREVENCIÓN DE FUGA DE DATOS (DATA LEAKAGE)
# ==============================================================================
GRUPO_CONGLOME = "conglome_25"  # Todo split y GroupKFold DEBE agrupar por este campo

# ==============================================================================
# VARIABLES OBJETIVO (TARGETS)
# ==============================================================================
TARGET_REZAGO = "rezago_escolar"            # Target principal supervisado (Binario)
TARGET_EXCLUSION_DIGITAL = "exclusion_digital"  # Target secundario supervisado (Binario)

# ==============================================================================
# DEFINICIÓN NORMATIVA DE EDADES SEGÚN MINEDU (Resolución Ministerial N° 064-2022)
# ==============================================================================
# Primaria: Grados 1 a 6 -> Edades normativas: 6, 7, 8, 9, 10, 11
# Secundaria: Grados 1 a 5 -> Edades normativas: 12, 13, 14, 15, 16
# Tolerancia normativa: +1 año (por corte de edad al 31 de marzo)
# Rezago Escolar = edad > (edad_normativa + 1)
TOLERANCIA_REZAGO = 1
