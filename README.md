# Vinculación de Entidades Clínicas — NER en Español → SNOMED CT

> Pipeline de PLN para vincular entidades médicas extraídas de textos clínicos en español con conceptos estandarizados de **SNOMED CT**, utilizando traducción automática neuronal, embeddings multilingües y búsqueda por similitud semántica.

---

## Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Presentación](#presentación)
- [Arquitectura del Pipeline](#arquitectura-del-pipeline)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Datos](#datos)
- [Uso](#uso)
- [Modelos](#modelos)
- [Evaluación](#evaluación)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Limitaciones Conocidas](#limitaciones-conocidas)

---

## Descripción General

Este notebook implementa un pipeline de **vinculación de entidades** (*entity linking*) para textos clínicos en español, evaluado sobre el corpus del reto compartido [DisTEMIST](https://temu.bsc.es/distemist/).

El punto de partida es un archivo `.tsv` que contiene entidades médicas ya extraídas de aproximadamente **750 documentos clínicos en español**, resultado de un proceso previo de Reconocimiento de Entidades Nombradas (NER) que no forma parte de este trabajo. Dicho archivo incluye, entre otros campos, la mención clínica en español y su código SNOMED CT, que es la información relevante para esta tarea.

El objetivo del pipeline es tomar cada mención clínica en español y resolverla hacia su **código de concepto SNOMED CT** más probable, sin necesidad de ajuste fino (*fine-tuning*) específico para la tarea. Para ello se apoya en:

1. Un diccionario de terminología médica con traducción manual (No verificado por un especialista).
2. Traducción automática neuronal como mecanismo de respaldo.
3. Similitud semántica mediante embeddings de oraciones multilingües.
4. Búsqueda eficiente por vectores con FAISS.

---

## Presentación

La presentación del proyecto está disponible directamente en este repositorio:

**[Vinculación de Entidades Médicas (PDF)](Presentation%20-%20Vinculacion%20de%20Entidades%20Medicas.pdf)**

Cubre el planteamiento del problema, las decisiones de diseño del pipeline y un análisis de los resultados de evaluación. Se recomienda revisarla antes de explorar el notebook.

---

## Arquitectura del Pipeline

```
Entidades NER en español (.tsv)
        │
        ▼
┌───────────────────────┐
│  Diccionario Médico   │  ← Traducciones manuales
│  (MEDICAL_DICT)       │
└──────────┬────────────┘
           │ menciones no cubiertas
           ▼
┌───────────────────────┐
│  Traducción Neuronal  │  ← Helsinki-NLP/opus-mt-es-en (MarianMT)
│  (ES → EN)            │    + correcciones de postprocesamiento
└──────────┬────────────┘
           │ menciones en inglés
           ▼
┌───────────────────────┐
│  Embeddings           │  ← intfloat/multilingual-e5-large
│  Multilingües         │    prefijo de consulta: "query:"
└──────────┬────────────┘
           │ vectores de consulta
           ▼
┌───────────────────────┐
│  Índice FAISS         │  ← IndexFlatIP (producto interno / coseno)
│  (Base SNOMED CT)     │    prefijo de pasaje: "passage:"
└──────────┬────────────┘
           │ top-K candidatos
           ▼
     Código SNOMED CT
```

---

## Requisitos

| Dependencia | Versión |
|---|---|
| Python | ≥ 3.9 |
| PyTorch | ≥ 2.0 |
| Transformers (HuggingFace) | ≥ 4.38 |
| FAISS | `faiss-cpu` o `faiss-gpu` |
| pandas | ≥ 2.0 |
| numpy | ≥ 1.24 |
| tqdm | ≥ 4.0 |

> **Nota sobre GPU:** El notebook detecta CUDA automáticamente. Se recomienda GPU para los pasos de traducción y generación de embeddings. La ejecución en CPU es posible pero considerablemente más lenta.

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/PLN-disca-iimas/clinical-linking-sp.git
cd clinical-linking-sp

# Instalar dependencias
pip install transformers accelerate torch faiss-cpu numpy pandas tqdm
```

Para FAISS con aceleración por GPU:

```bash
pip install faiss-gpu
```

---

## Datos

> **El snapshot de SNOMED CT NO está incluido en este repositorio** — es un archivo licenciado que no puede redistribuirse. El archivo `.tsv` de DisTEMIST sí está incluido en `data/`. Consulta las instrucciones para obtener el snapshot de SNOMED CT.

### Entrada — Entidades NER

```
data/distemist_subtrack2_training1_linking.tsv
```

Archivo TSV resultado del proceso de Reconocimiento de Entidades Nombradas (NER). Las columnas relevantes para este pipeline son:

| Columna | Descripción |
|---|---|
| `span` | Mención clínica en español (ej. `"fractura conminuta"`) |
| `code` | Código SNOMED CT de referencia (*gold standard*) |

Este archivo forma parte del corpus **DisTEMIST**. También puede descargarse desde la [página oficial del reto](https://temu.bsc.es/distemist/).

### Entrada — Snapshot de SNOMED CT (requiere licencia)

```
sct2_Description_Snapshot-en_INT_20260101.txt
```

Snapshot oficial de la Edición Internacional de SNOMED CT. Este archivo está **excluido del repositorio** por dos razones: supera el límite de 100 MB de GitHub y su redistribución no está permitida bajo la licencia de SNOMED CT.

**Cómo obtenerlo:**
1. Registrarse para obtener una licencia gratuita en [SNOMED International](https://www.snomed.org/get-snomed).
2. Descargar el paquete de la Edición Internacional.
3. Localizar el archivo `sct2_Description_Snapshot-en_INT_<fecha_release>.txt` dentro del paquete.
4. Colocarlo en `/content/` (Colab) o actualizar la variable `path_snomed` en el notebook.

---

## Uso

### Google Colab (recomendado)

1. Abrir `Clinical_coding.ipynb` en Google Colab.
2. Subir `sct2_Description_Snapshot-en_INT_20260101.txt` a `/content/`.
3. Ejecutar todas las celdas en orden.

### Ejecución local

Actualizar las variables de ruta al inicio de las celdas de carga de datos:

```python
path_train  = "/ruta/a/distemist_subtrack2_training1_linking.tsv"
path_snomed = "/ruta/a/sct2_Description_Snapshot-en_INT_20260101.txt"
```

Luego ejecutar el notebook celda por celda o mediante:

```bash
jupyter nbconvert --to notebook --execute Clinical_coding.ipynb
```

---

## Modelos

| Modelo | Función | Fuente |
|---|---|---|
| `Helsinki-NLP/opus-mt-es-en` | Traducción español → inglés | [HuggingFace Hub](https://huggingface.co/Helsinki-NLP/opus-mt-es-en) |
| `intfloat/multilingual-e5-large` | Embeddings de oraciones (consulta y pasaje) | [HuggingFace Hub](https://huggingface.co/intfloat/multilingual-e5-large) |

Ambos modelos se descargan automáticamente en la primera ejecución.

---

## Evaluación

El pipeline se evalúa con métricas estándar de **recall@K** sobre el subconjunto de vinculación de DisTEMIST:

| Métrica | Descripción |
|---|---|
| **Exactitud / Recall@1** | El código SNOMED correcto es la predicción de mayor rango |
| **Recall@5** | El código correcto aparece entre los 5 candidatos principales |
| **Recall@10** | El código correcto aparece entre los 10 candidatos principales |

Los resultados se imprimen tras el paso de búsqueda FAISS. También se genera un DataFrame completo (`df_preds`) con el desglose por entidad, incluyendo las banderas `match`, `in_top5` e `in_top10` para análisis de errores.

---

## Estructura del Proyecto

```
.
├── Clinical_coding.ipynb                                # Notebook principal del pipeline
├── Presentation - Vinculacion de Entidades Medicas.pdf  # Presentación del proyecto
├── README.md                                            # Este archivo
├── .gitignore                                           # Excluye el snapshot de SNOMED CT
└── data/
    └── distemist_subtrack2_training1_linking.tsv        # Entidades NER de DisTEMIST (incluido)
    # sct2_Description_Snapshot-en_INT_*.txt             # ← NO incluido (licencia + tamaño)
```

---

## Limitaciones Conocidas

- **Cobertura del diccionario:** `MEDICAL_DICT` cubre un conjunto de términos frecuentes. Las expresiones poco comunes o muy especializadas caen al modelo de traducción neuronal, que puede introducir ruido. Cabe señalar que las traducciones no fueron revisadas por ningún experto en el área.
- **Submuestreo de SNOMED CT:** Para mantener el índice FAISS manejable, los conceptos que no son objetivo se muestrean de forma estratificada a 3,000 por etiqueta semántica. Esto mejora la velocidad pero puede reducir el recall para conceptos en etiquetas poco representadas.
- **Sin ajuste fino:** El modelo de embeddings se usa de forma *zero-shot*. Se espera que el ajuste fino específico sobre datos de DisTEMIST mejore sustancialmente el Recall@1.
- **Granularidad semántica en SNOMED CT:** Un mismo código de concepto puede tener múltiples descripciones clínicas distintas en la base de conocimiento (por ejemplo, el código ´160602000´ agrupa "Occasional smoker", "Trivial smoker - < 1 cig/day" y otras variantes). Dado que el pipeline aplica `drop_duplicates` por conceptId` al construir el índice FAISS, solo una de esas descripciones queda representada en el espacio vectorial. Si la descripción retenida no es la más cercana semánticamente a la mención en español del .tsv, el modelo no recuperará el código correcto aunque el concepto exista en la base, lo que penaliza artificialmente las métricas de evaluación.
