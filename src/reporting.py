"""Generación reproducible del informe científico final en PDF (seis páginas)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

from . import config


OUT = config.ROOT_DIR / "output" / "pdf" / "paper_brecha_digital_ieee.pdf"
W, H = A4
M = 38
BLUE = colors.HexColor("#123B5D")
CYAN = colors.HexColor("#12A4B4")
LIGHT = colors.HexColor("#EAF2F6")
TEXT = colors.HexColor("#18242D")


STYLES = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=19, leading=22, alignment=TA_CENTER, textColor=BLUE),
    "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=9, leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#52636F")),
    "abstract": ParagraphStyle("abstract", fontName="Helvetica", fontSize=8.2, leading=10.5, alignment=TA_JUSTIFY, textColor=TEXT),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=BLUE, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=8.1, leading=10.4, alignment=TA_JUSTIFY, textColor=TEXT),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=7.1, leading=8.7, alignment=TA_LEFT, textColor=TEXT),
    "caption": ParagraphStyle("caption", fontName="Helvetica-Oblique", fontSize=7, leading=8, alignment=TA_CENTER, textColor=colors.HexColor("#4F5F68")),
}


def _header_footer(c: canvas.Canvas, page: int) -> None:
    c.setFillColor(BLUE)
    c.rect(0, H - 24, W, 24, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 7.5)
    c.drawString(M, H - 16, "PROYECTO IA | ENAHO - ESCALE 2025")
    c.setStrokeColor(CYAN)
    c.setLineWidth(1.2)
    c.line(M, 26, W - M, 26)
    c.setFillColor(colors.HexColor("#52636F"))
    c.setFont("Helvetica", 7)
    c.drawString(M, 15, "Unidad de Inteligencia Artificial - Ingeniería de Sistemas")
    c.drawRightString(W - M, 15, f"Página {page} de 6")


def _paragraph(c, text, style, x, y_top, width, height=300):
    p = Paragraph(text, style)
    _, h = p.wrap(width, height)
    p.drawOn(c, x, y_top - h)
    return y_top - h


def _section(c, title, x, y, width):
    c.setFillColor(LIGHT)
    c.roundRect(x, y - 17, width, 17, 3, fill=1, stroke=0)
    c.setFillColor(BLUE)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x + 6, y - 12, title)
    return y - 23


def _image(c, path: Path, x, y_top, width, height):
    if not path.exists():
        return
    img = ImageReader(str(path))
    iw, ih = img.getSize()
    scale = min(width / iw, height / ih)
    dw, dh = iw * scale, ih * scale
    c.drawImage(img, x + (width - dw) / 2, y_top - dh, dw, dh, preserveAspectRatio=True, mask="auto")


def _table(c, data, x, y_top, widths, font=6.6, header=True):
    wrapped = [[Paragraph(str(v), STYLES["small"]) for v in row] for row in data]
    t = Table(wrapped, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AFC0CB")),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), BLUE), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    for row in range(1 if header else 0, len(data)):
        if row % 2 == 0:
            style.append(("BACKGROUND", (0, row), (-1, row), colors.HexColor("#F5F8FA")))
    t.setStyle(TableStyle(style))
    _, h = t.wrap(sum(widths), 500)
    t.drawOn(c, x, y_top - h)
    return y_top - h


def generar_pdf(output: Path = OUT) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    figures = config.FIGURES_DIR
    cv = pd.read_csv(config.REPORTS_DIR / "modelos_cv_resumen.csv").set_index("modelo")
    test = pd.read_csv(config.REPORTS_DIR / "modelos_metricas_test.csv").set_index("modelo")
    profiles = pd.read_csv(config.REPORTS_DIR / "perfiles_clusters.csv")
    c = canvas.Canvas(str(output), pagesize=A4)
    c.setTitle("Brecha digital y rezago escolar en el Perú - ENAHO ESCALE 2025")
    c.setAuthor("Unidad de Inteligencia Artificial - Ingeniería de Sistemas")
    c.setSubject("Clustering, clasificación supervisada y explicabilidad SHAP con ENAHO-ESCALE 2025")
    c.setKeywords("ENAHO, ESCALE, rezago escolar, brecha digital, CatBoost, SHAP")

    # Página 1
    _header_footer(c, 1)
    y = H - 56
    y = _paragraph(c, "Brecha digital y rezago escolar en el Perú: integración ENAHO-ESCALE 2025 con aprendizaje automático explicable", STYLES["title"], M + 20, y, W - 2 * M - 40)
    y -= 8
    y = _paragraph(c, "Unidad de Inteligencia Artificial | Escuela Profesional de Ingeniería de Sistemas | Octubre de 2026", STYLES["subtitle"], M, y, W - 2 * M)
    y -= 18
    y = _section(c, "RESUMEN", M, y, W - 2 * M)
    abstract = ("Se desarrolló un pipeline reproducible para caracterizar la brecha digital y predecir rezago escolar en estudiantes peruanos de 6 a 17 años. Se fusionaron ENAHO 2025 y ESCALE a nivel de persona, hogar y distrito. El dataset final contiene <b>23,668 estudiantes</b>; 9,382 fueron evaluables y 520 presentaron rezago (5.54%). La validación agrupada por conglomerado evitó contaminación territorial. K-Means identificó dos perfiles; CatBoost obtuvo el mayor PR-AUC medio en GroupKFold (0.0937). SHAP destacó dominio geográfico, oferta escolar, tableta, ratio alumno/docente y recursos económicos. Los resultados son predictivos y asociativos, no causales.")
    y = _paragraph(c, abstract, STYLES["abstract"], M, y, W - 2 * M)
    y -= 9
    _paragraph(c, "<b>Palabras clave:</b> ENAHO, ESCALE, rezago escolar, brecha digital, GroupKFold, CatBoost, SHAP.", STYLES["small"], M, y, W - 2 * M)
    y -= 32
    y = _section(c, "I. INTRODUCCIÓN Y CONTRIBUCIÓN", M, y, W - 2 * M)
    intro = ("La desigualdad educativa combina restricciones del hogar, acceso tecnológico y heterogeneidad territorial. La guía del proyecto exige integrar fuentes oficiales, descubrir perfiles, comparar clasificadores y auditar decisiones [1]. La estructura muestral ENAHO usa conglomerados como unidades primarias [2]; por ello una partición aleatoria simple inflaría el rendimiento. Este trabajo aporta una integración trazable, un protocolo de cero fuga, comparación de seis modelos y explicaciones globales e individuales.")
    y = _paragraph(c, intro, STYLES["body"], M, y, W - 2 * M)
    y -= 16
    c.setFont("Helvetica-Bold", 8); c.setFillColor(BLUE)
    labels = ["ENAHO\npersona-hogar", "ESCALE\ndistrito", "Clustering\nPCA-UMAP", "Clasificación\nGroupKFold", "SHAP\nglobal-local"]
    gap = 10; boxw = (W - 2*M - 4*gap) / 5; x = M
    for i, label in enumerate(labels):
        c.setFillColor(LIGHT); c.roundRect(x, 112, boxw, 48, 5, fill=1, stroke=0)
        c.setFillColor(BLUE); c.setFont("Helvetica-Bold", 7.5)
        for j, line in enumerate(label.split("\n")):
            tw = stringWidth(line, "Helvetica-Bold", 7.5)
            c.drawString(x + (boxw - tw)/2, 140 - j*11, line)
        if i < 4:
            c.setStrokeColor(CYAN); c.line(x + boxw + 2, 136, x + boxw + gap - 2, 136)
        x += boxw + gap
    _header_footer(c, 1)
    c.showPage()

    # Página 2
    _header_footer(c, 2); y = H - 48
    y = _section(c, "II. DATOS, TARGET Y PROTOCOLO ANTI-FUGA", M, y, W - 2*M)
    y = _paragraph(c, "Se integraron M100, M200, M300 y Sumaria de ENAHO con el Padrón de Instituciones y Locales ESCALE. Todas las llaves se preservaron como texto y los UBIGEO se normalizaron a seis dígitos. La población se restringió después del cruce a edades de 6 a 17 años.", STYLES["body"], M, y, W - 2*M)
    y -= 8
    stats = [["Indicador", "Valor"], ["Estudiantes 6-17", "23,668"], ["Distritos", "1,240"], ["Conglomerados", "5,050"], ["Evaluables para rezago", "9,382"], ["Rezago positivo", "520 (5.54%)"], ["Exclusión digital", "332 (1.40%)"]]
    y = _table(c, stats, M, y, [210, 120])
    y -= 11
    y = _section(c, "Definición y controles", M, y, W - 2*M)
    body = ("El target se definió como <b>1[edad &gt; edad normativa(nivel, grado) + 1]</b> para estudiantes matriculados o asistentes en primaria/secundaria. Se excluyeron edad, nivel, grado, edad normativa y población evaluable de los predictores. La partición externa fue GroupShuffleSplit 80/20; la validación interna, GroupKFold(5). Imputación, RobustScaler y one-hot se ajustaron dentro de cada fold. Los 3,243 conglomerados de train y 811 de test tuvieron solapamiento cero.")
    y = _paragraph(c, body, STYLES["body"], M, y, W - 2*M)
    y -= 12
    controls = [["Riesgo", "Control implementado"], ["Pérdida de ceros", "dtype string + zfill(6)"], ["Duplicidad relacional", "validate one_to_one / many_to_one"], ["Contaminación espacial", "Split y CV por CONGLOME"], ["Fuga del target", "Exclusión de componentes directos"], ["Desbalance", "Pesos de clase + factor07 normalizado"]]
    y = _table(c, controls, M, y, [165, 305])
    y -= 12
    y = _section(c, "Variables de modelado", M, y, W - 2*M)
    _paragraph(c, "<b>Hogar:</b> gasto, ingreso, electricidad, computadora, laptop, tableta e internet. <b>Persona:</b> sexo, lengua, uso reciente de internet y equipamiento. <b>ESCALE:</b> ratio alumno/docente, locales, gestión pública y ruralidad. Las transformaciones logarítmicas se aplicaron a gasto e ingreso per cápita.", STYLES["body"], M, y, W - 2*M)
    _header_footer(c, 2)
    c.showPage()

    # Página 3
    _header_footer(c, 3); y = H - 48
    y = _section(c, "III. RESULTADOS NO SUPERVISADOS", M, y, W - 2*M)
    y = _paragraph(c, "K-Means y Gaussian Mixture se compararon sobre diez variables sin incluir el target. El balance de Silhouette, Davies-Bouldin y Calinski-Harabasz seleccionó k=2. Las tres primeras componentes PCA explicaron 81.21% de la varianza.", STYLES["body"], M, y, W - 2*M)
    y -= 6
    metrics = [["k", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz"], ["2", "0.3233", "1.2251", "1769.0"], ["3", "0.2364", "1.4203", "1359.2"], ["4", "0.2177", "1.4583", "1174.5"]]
    y = _table(c, metrics, M, y, [55, 110, 130, 165])
    y -= 8
    _image(c, figures / "clustering_metricas_k.png", M, y, 235, 185)
    _image(c, figures / "pca_clusters.png", M + 245, y, 275, 185)
    y -= 193
    profile_data = [["Perfil", "n", "Internet", "Uso internet", "Rezago", "Exclusión digital"]]
    for _, row in profiles.iterrows():
        profile_data.append([row["nombre_cluster"], f"{int(row['n']):,}", f"{row['internet_hogar']:.1%}", f"{row['uso_internet_mes']:.1%}", f"{row['tasa_rezago']:.2%}", f"{row['tasa_exclusion_digital']:.2%}"])
    y = _table(c, profile_data, M, y, [185, 48, 64, 70, 55, 74])
    y -= 8
    _paragraph(c, "El perfil rural vulnerable presenta menor conectividad y una tasa de exclusión digital casi 38 veces mayor. La tasa de rezago supera en 1.12 puntos porcentuales a la del perfil urbano. El target se utilizó solo después del ajuste para caracterización externa.", STYLES["body"], M, y, W - 2*M)
    _header_footer(c, 3)
    c.showPage()

    # Página 4
    _header_footer(c, 4); y = H - 48
    y = _section(c, "IV. CLASIFICACIÓN SUPERVISADA", M, y, W - 2*M)
    y = _paragraph(c, "El criterio de selección fue PR-AUC media en GroupKFold, apropiado para una prevalencia de 5.54%. CatBoost fue el campeón de validación; XGBoost obtuvo el mayor PR-AUC puntual en test y la regresión logística el mayor recall.", STYLES["body"], M, y, W - 2*M)
    rows = [["Modelo", "PR-AUC CV", "ROC test", "PR test", "F1 macro", "Recall", "Brier"]]
    order = ["CatBoost", "Random Forest", "XGBoost", "Regresion Logistica", "LightGBM", "Arbol CART"]
    for name in order:
        rows.append([name, f"{cv.loc[name,'pr_auc_media']:.4f}", f"{test.loc[name,'roc_auc']:.4f}", f"{test.loc[name,'pr_auc']:.4f}", f"{test.loc[name,'f1_macro']:.4f}", f"{test.loc[name,'recall']:.3f}", f"{test.loc[name,'brier']:.4f}"])
    y = _table(c, rows, M, y - 7, [112, 70, 65, 65, 68, 55, 58])
    y -= 10
    _image(c, figures / "modelos_curvas_roc_pr.png", M, y, 320, 255)
    _image(c, figures / "modelo_campeon_matriz_confusion.png", M + 330, y, 190, 210)
    y -= 260
    y = _section(c, "Análisis de errores", M, y, W - 2*M)
    _paragraph(c, "CatBoost produjo 1,316 verdaderos negativos, 483 falsos positivos, 77 falsos negativos y 48 verdaderos positivos. El balanceo mejora detección, pero reduce precisión y distorsiona probabilidades. El desempeño es moderado: el modelo sirve para priorización exploratoria, no para decisiones automáticas. Se requiere calibración antes de un uso operativo.", STYLES["body"], M, y, W - 2*M)
    _header_footer(c, 4)
    c.showPage()

    # Página 5
    _header_footer(c, 5); y = H - 48
    y = _section(c, "V. EXPLICABILIDAD ALGORÍTMICA CON SHAP", M, y, W - 2*M)
    y = _paragraph(c, "TreeExplainer auditó CatBoost sobre 1,924 casos de prueba. Los valores SHAP están en log-odds: valores positivos elevan la salida de riesgo y negativos la reducen.", STYLES["body"], M, y, W - 2*M)
    _image(c, figures / "shap_summary_beeswarm.png", M, y - 5, W - 2*M, 390)
    y -= 398
    _image(c, figures / "shap_dependencia_lengua_internet.png", M, y, 250, 165)
    _image(c, figures / "shap_dependencia_gasto_ratio.png", M + 265, y, 250, 165)
    y -= 172
    _paragraph(c, "Dominio geográfico, oferta de locales, tableta, ratio alumno/docente y recursos económicos encabezan la importancia. La relación gasto-riesgo es no lineal; la señal lengua-internet es débil y no permite afirmar un efecto multiplicador fuerte. SHAP explica asociaciones internas del modelo, no causalidad.", STYLES["body"], M, y, W - 2*M)
    _header_footer(c, 5)
    c.showPage()

    # Página 6
    _header_footer(c, 6); y = H - 48
    y = _section(c, "VI. AUDITORÍA INDIVIDUAL, RECOMENDACIONES Y CONCLUSIONES", M, y, W - 2*M)
    _image(c, figures / "shap_waterfall_alto_riesgo.png", M, y, 250, 255)
    _image(c, figures / "shap_waterfall_bajo_riesgo.png", M + 265, y, 250, 255)
    y -= 262
    y = _paragraph(c, "El caso positivo extremo alcanzó P(rezago)=0.828; el alto ratio alumno/docente, dominio territorial, lengua originaria y ruralidad aumentaron el riesgo. El caso negativo extremo obtuvo 0.006; equipamiento TIC, computadora, tableta y mejores recursos económicos actuaron como factores protectores dentro del modelo.", STYLES["body"], M, y, W - 2*M)
    y -= 8
    y = _section(c, "Recomendaciones de política", M, y, W - 2*M)
    y = _paragraph(c, "1) Priorizar acompañamiento en conglomerados rurales vulnerables. 2) Vincular entrega de dispositivos con uso efectivo, electricidad y soporte. 3) Monitorear distritos con ratios alumno/docente altos. 4) Usar el modelo como apoyo humano y auditable, nunca como decisión automática sobre estudiantes.", STYLES["body"], M, y, W - 2*M)
    y -= 8
    y = _section(c, "Limitaciones", M, y, W - 2*M)
    y = _paragraph(c, "El corte es transversal; la clase positiva es escasa; ESCALE se agregó por distrito; y el rendimiento predictivo es moderado. Se requieren calibración, validación temporal/geográfica y auditorías de equidad por sexo, lengua y territorio.", STYLES["body"], M, y, W - 2*M)
    y -= 8
    y = _section(c, "Referencias", M, y, W - 2*M)
    refs = ("[1] Lab de IA, Guía de Evaluación: Brecha Digital y Educación 2025. [2] INEI, Ficha Técnica Panel ENAHO 2021-2025. [3] MINEDU, RM 064-2022-MINEDU. [4] INEI, Cuestionarios ENAHO M100 y M300, 2025. [5] MINEDU-ESCALE, Diccionario del Censo Educativo 2025. [6] Lundberg y Lee, A Unified Approach to Interpreting Model Predictions, NeurIPS, 2017. [7] Prokhorenkova et al., CatBoost, NeurIPS, 2018.")
    _paragraph(c, refs, STYLES["small"], M, y, W - 2*M)
    _header_footer(c, 6)
    c.save()
    return output


if __name__ == "__main__":
    print(generar_pdf())
