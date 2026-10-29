# Brecha Digital y Rezago Escolar en el Perú: Integración ENAHO-ESCALE 2025 con IA Explicable

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Academic-green.svg)]()
[![Dataset](https://img.shields.io/badge/data-ENAHO%20%7C%20ESCALE%202025-orange.svg)]()
[![Model](https://img.shields.io/badge/champion-CatBoost-yellow.svg)]()
[![XAI](https://img.shields.io/badge/interpretability-SHAP-purple.svg)]()

> **Proyecto de Evaluación — Laboratorio de Inteligencia Artificial**  
> **Escuela Profesional de Ingeniería de Sistemas**  
> **Corte de Datos:** ENAHO (INEI) + ESCALE (MINEDU) 2025  
> **Semilla Global de Reproducibilidad:** `random_state = 42`

---

## 📌 Resumen del Proyecto

Este repositorio contiene el pipeline completo, reproducible y auditable de **Machine Learning (No Supervisado + Supervisado + Explicabilidad XAI con SHAP)** que analiza el impacto de la brecha digital y las condiciones socioeconómicas sobre el **rezago escolar** en estudiantes peruanos de **6 a 17 años**.

El estudio integra dos fuentes oficiales del Estado Peruano para el año 2025:
1. **ENAHO 2025 (INEI - Encuesta 1041):** Microdatos de vivienda (M100), miembros del hogar (M200), educación y TICs (M300) y condiciones socioeconómicas/pobreza (Sumaria).
2. **ESCALE 2025 (MINEDU):** Padrón de Instituciones Educativas y Padrón de Locales Escolares (ratios de alumnos por docente, proporción de colegios públicos, infraestructura y ruralidad distrital).

---

## 🏗️ Estructura del Repositorio

```text
├── data/
│   ├── raw/                             # Indicaciones para datos crudos INEI/ESCALE
│   │   ├── README_DATOS_CRUDOS.md
│   │   └── .gitkeep
│   ├── processed/                       # Datasets limpios finales listos para ML
│   │   ├── dataset_enaho_escale_2025.parquet  (1.1 MB - Optimizado)
│   │   ├── dataset_enaho_escale_2025.csv      (7.1 MB)
│   │   └── diccionario_variables.csv          (50 variables descritas al detalle)
│   └── docs/                            # Diccionarios y cuestionarios oficiales (PDF)
│       ├── Guia_de_Evaluacion.pdf
│       ├── Ficha_Tecnica_ENAHO_1041.pdf
│       ├── CUESTIONARIO_M300_educacion_TIC_2025.pdf
│       └── Diccionario_ENAHO_panel.pdf
├── notebooks/                           # 4 Jupyter Notebooks secuenciales y reproducibles
│   ├── 01_ingenieria_datos.ipynb        # Ingesta, merge relacional y protocolo anti-fuga
│   ├── 02_clustering_perfilamiento.ipynb # Preprocesamiento, PCA, UMAP y métricas k
│   ├── 03_modelos_supervisados.ipynb    # GroupKFold, 6 clasificadores, curvas ROC/PR
│   └── 04_explicabilidad_shap.ipynb     # SHAP Beeswarm, Dependence y Waterfall plots
├── src/                                 # Módulos Python limpios y desacoplados
│   ├── config.py                        # Parámetros, rutas, constantes y llaves
│   ├── data_loader.py                   # Carga robusta con tipado estricto (dtype=str)
│   ├── feature_engineering.py           # Agregación distrital ESCALE y merge maestro
│   ├── clustering.py                    # Algoritmos K-Means, GMM, PCA, UMAP
│   ├── supervised.py                    # Modelado supervisado con GroupKFold
│   ├── evaluation.py                    # Métricas de clasificación y curvas de calibración
│   ├── explainability.py                # XAI con SHAP TreeExplainer
│   └── reporting.py                     # Compilador automatizado del paper en PDF
├── reports/                             # Informe científico y artefactos de evaluación
│   ├── paper_brecha_digital_ieee.pdf    # Artículo científico de 6 páginas en formato IEEE
│   ├── paper_brecha_digital_ieee.md     # Manuscrito científico en Markdown
│   ├── figures/                         # 11 Gráficos de alta resolución (PNG)
│   ├── models/                          # Modelos entrenados serializados (.joblib)
│   └── *.csv / *.json                   # Tablas de métricas por fold y metadatos
├── requirements.txt                     # Dependencias exactas del proyecto
└── README.md                            # Documentación principal del repositorio
```

---

## 🛡️ Protocolo Anti-Fuga de Datos (Zero Data Leakage)

Para evitar la contaminación espacial y de conglomerado entre el conjunto de entrenamiento y prueba:
- **Partición por Conglomerado:** Se utilizó `GroupShuffleSplit` (80% train / 20% test) y `GroupKFold(n_splits=5)` agrupando estrictamente por `conglome_25`. Esto garantiza que los 3,243 conglomerados de entrenamiento y los 811 de prueba sean disjuntos (**0 conglomerados compartidos**).
- **Transformaciones Aisladas:** Todo escalador (`RobustScaler`), imputador (`SimpleImputer`) y codificador (`OneHotEncoder`) se ajusta (`fit`) **exclusivamente dentro de los folds de entrenamiento** mediante `Pipeline` de scikit-learn.
- **Exclusión de Proxies Directos:** Se eliminaron como variables predictoras la edad (`p208a_25`), nivel educativo (`p308a_25`), grado actual (`p308b_25`) y edad normativa calculada, evitando cualquier circularidad matemática hacia el target.

---

## 📊 Principales Resultados

### 1. Perfilamiento No Supervisado (Clustering, $k=2$)
K-Means fue seleccionado sobre Gaussian Mixture Models por exhibir mejor balance métrico (**Silhouette = 0.3233**, **Davies-Bouldin = 1.2251**, **Calinski-Harabasz = 1769.0**). La proyección PCA explicó el **81.21% de la varianza** en 3 componentes.

| Perfil Socio-Tecnológico | Muestra ($n$) | Internet Hogar | Uso Internet | Ratio Alumno/Docente | Tasa Rezago Observada |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Rural vulnerable con aislamiento digital y dependencia pública** | 10,239 | 91.49% | 58.70% | 12.38 | **6.19%** |
| **Urbano consolidado con alta conectividad y mejores recursos** | 13,429 | 98.01% | 83.16% | 17.43 | **5.07%** |

*(Nota metodológica: El rezago escolar se midió como variable de contraste externa posterior, evitando circularidad en la formación de clusters).*

### 2. Modelado Supervisado (Predicción de Rezago Escolar)
Ante un problema con fuerte desbalance de clases (prevalencia positiva del **5.54%**), el criterio de selección fue el **PR-AUC media en GroupKFold de 5 folds**:

| Modelo | PR-AUC CV (Media ± Desv) | ROC-AUC Test | PR-AUC Test | F1-Macro | Recall | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CatBoost (Campeón)** | **0.0937 ± 0.0178** | 0.5969 | 0.1048 | 0.4855 | 0.3840 | 0.1803 |
| **Random Forest** | 0.0912 ± 0.0198 | 0.5973 | 0.0939 | 0.5170 | 0.0560 | **0.0941** |
| **XGBoost** | 0.0887 ± 0.0137 | 0.5951 | **0.1068** | 0.4904 | 0.3200 | 0.1677 |
| **Regresión Logística (Baseline)** | 0.0846 ± 0.0261 | **0.6196** | 0.1055 | 0.4046 | **0.6720** | 0.2595 |
| **LightGBM** | 0.0844 ± 0.0177 | 0.5637 | 0.0902 | **0.5247** | 0.2160 | 0.1280 |
| **Árbol CART (Baseline)** | 0.0656 ± 0.0020 | 0.6001 | 0.0820 | 0.5092 | 0.3280 | 0.2377 |

### 3. Explicabilidad Algorítmica con SHAP
Aplicando `shap.TreeExplainer` sobre CatBoost en el conjunto de prueba ($n=1,924$):
- **Top 5 Determinantes Globales:** Dominio geográfico regional (Sierra Sur / Selva), cantidad distrital de locales escolares, tenencia de tablet en el hogar (`p1143_25`), ratio alumno/docente distrital y gasto per cápita del hogar.
- **Casos Extremos Auditados:**
  - *Estudiante de Alto Riesgo ($P = 0.828$):* Rezago predicho correctamente; impacto aditivo crítico de ruralidad extrema, lengua materna originaria y sobrecarga docente.
  - *Estudiante de Bajo Riesgo ($P = 0.006$):* Sin rezago; efecto protector de equipamiento digital múltiple (computadora + tableta) y suficiencia económica familiar.

---

## 🚀 Guía de Reproducción Rápida

### 1. Clonar el Repositorio y Configurar Entorno
```bash
git clone <URL_DEL_REPOSITORIO>
cd entrega_github_proyecto_ia

# Crear entorno virtual
python -m venv .venv

# Activar entorno (Windows PowerShell)
.venv\Scripts\Activate.ps1
# O en Linux/macOS: source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Ejecutar los Notebooks
Puedes abrir y ejecutar secuencialmente los 4 notebooks:
```bash
jupyter lab notebooks/
```
Los notebooks ya contienen los outputs y figuras generadas con `random_state = 42`.

### 3. Re-entrenar Modelos desde Consola (Opcional)
```bash
python -m src.supervised
python -m src.explainability
```

---

## 📄 Informe Científico Final

El artículo científico completo de 6 páginas en formato IEEE de dos columnas se encuentra disponible en:
- **PDF Listo para Lectura/Impresión:** [`reports/paper_brecha_digital_ieee.pdf`](reports/paper_brecha_digital_ieee.pdf)
- **Fuente Markdown:** [`reports/paper_brecha_digital_ieee.md`](reports/paper_brecha_digital_ieee.md)

---

## 📚 Referencias Principales

1. **INEI (2025).** *Ficha Técnica de la Base de Datos Panel ENAHO 2021-2025*. Instituto Nacional de Estadística e Informática, Lima, Perú.
2. **MINEDU (2022).** *Resolución Ministerial N.° 064-2022-MINEDU: Disposiciones para el inicio del año escolar en las instituciones educativas y programas educativos de la Educación Básica*. Ministerio de Educación del Perú.
3. **Lundberg, S. M., & Lee, S.-I. (2017).** *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems (NeurIPS 2017).
4. **Prokhorenkova, L., et al. (2018).** *CatBoost: Unbiased Boosting with Categorical Features*. Advances in Neural Information Processing Systems (NeurIPS 2018).
