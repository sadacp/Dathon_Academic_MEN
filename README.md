# Datathon-Academico-
# 📊 Más allá de los números: desigualdades educativas en Colombia (2011–2023)

**Proyecto de análisis de brechas educativas municipales con proyecciones hasta 2026, en línea con el ODS 4, el ODS 10 y el PNDE 2016–2026.**

---

## 📌 Descripción del proyecto

Este proyecto analiza la evolución de indicadores clave de la educación básica primaria, secundaria y media en los municipios de Colombia entre **2011 y 2023**, e identifica **brechas territoriales persistentes** en:

- Cobertura neta
- Tasa de deserción
- Aprobación
- Reprobación
- Repitencia

Con base en estos datos, se desarrollaron y compararon varios modelos de predicción para proyectar los indicadores hasta el año **2026**, con el objetivo de anticipar escenarios futuros y formular **recomendaciones orientadas a reducir las desigualdades educativas**, especialmente en municipios con mayor rezago.

---

## 🎯 Pregunta problema

> ¿Cuáles son las principales brechas en cobertura, deserción, aprobación, reprobación y repitencia en la educación básica primaria, secundaria y media entre los municipios de Colombia (2011–2023), y qué factores explicativos podrían sustentar recomendaciones orientadas a reducir dichas brechas, en línea con el ODS 4, el ODS 10 y el PNDE 2016–2026?

---

## 🛠️ Herramientas y tecnologías utilizadas

- **Python**
  - pandas, numpy
  - matplotlib, seaborn, plotly
  - statsmodels (ARIMA)
  - fbprophet (Prophet)
  - scikit-learn (regresión lineal y polinómica, clustering)
  - folium (visualización geográfica)
- **Jupyter Notebook**
- **GeoJSON para mapas municipales**
- **Datos oficiales del Ministerio de Educación Nacional de Colombia**

---

## 🧪 Modelos explorados

Durante el desarrollo del proyecto se entrenaron distintos modelos para realizar predicciones de los indicadores educativos:

| Modelo                  | Descripción breve                                                |
|-------------------------|------------------------------------------------------------------|
| **ARIMA**               | Modelo autorregresivo integrado con media móvil para series temporales. |
| **Prophet**             | Modelo aditivo desarrollado por Meta (Facebook), ideal para datos con estacionalidad. |
| **Regresión lineal**    | Modelo simple de tendencia lineal a lo largo del tiempo.         |
| **Regresión polinómica**| Extensión de la regresión lineal para ajustar relaciones no lineales. |

📌 **Se eligió ARIMA** como modelo final para la presentación debido a su buen ajuste a los datos, interpretabilidad y rendimiento para proyecciones a corto y mediano plazo.

---

## 📈 Metodología

1. **Carga y limpieza de datos** (2011–2023) por municipio y nivel educativo.
2. **Análisis exploratorio de datos (EDA)** para identificar brechas significativas.
3. **Comparación de modelos de predicción**: ARIMA, Prophet, regresión lineal y polinómica.
4. **Modelado final con ARIMA** para proyecciones hasta 2026.
5. **Clustering de municipios** para detectar grupos con características educativas similares.
6. **Visualización interactiva y mapas georreferenciados** para evidenciar desigualdades educativas.
7. **Formulación de recomendaciones** en línea con políticas públicas (ODS, PNDE).

---

## 🔍 Principales hallazgos

- Persisten brechas significativas entre municipios en todos los indicadores educativos.
- Municipios con menor cobertura presentan mayores tasas de deserción y repitencia.
- Las proyecciones ARIMA indican que, sin intervenciones, las desigualdades podrían mantenerse o incluso aumentar hasta 2026.
- El análisis de clústeres permitió identificar **territorios prioritarios** para la intervención educativa focalizada.

---

## 🌍 Enfoque de política pública

Este proyecto busca aportar evidencia para el **diseño de políticas focalizadas y sostenibles** que contribuyan a la **equidad territorial en la educación**, alineándose con:

- 📘 **ODS 4:** Educación de calidad.
- ⚖️ **ODS 10:** Reducción de las desigualdades.
- 📑 **PNDE 2016–2026:** Políticas para mejorar la calidad, cobertura y permanencia escolar.

