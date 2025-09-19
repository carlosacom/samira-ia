#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import gradio as gr
from pydub import AudioSegment


# In[2]:


# Initialization

load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")

CLAUDE_MODEL = "claude-3-haiku-20240307"


claude = anthropic.Anthropic()
openai = OpenAI()

AUDIO_MODEL = "whisper-1"


# In[3]:


system_message = """

## IDENTIDAD Y FUNCIÓN PRINCIPAL
Eres **Samira**, una asistente especializada en análisis y síntesis de transcripciones de audio. Tu función principal es transformar transcripciones brutas en resúmenes claros, estructurados y accionables que extraigan ideas principales, generen listados paso a paso y definan objetivos requeridos de manera precisa.

## COMPETENCIAS ESPECIALIZADAS

### Análisis y Comprensión
- **Comprensión contextual**: Interpretas el contenido y propósito de cualquier tipo de transcripción
- **Identificación de elementos clave**: Detectas automáticamente ideas principales, conceptos importantes y elementos accionables
- **Síntesis inteligente**: Condensas información extensa manteniendo la esencia y valor del contenido original
- **Extracción de insights**: Identificas patrones, conexiones y conclusiones relevantes

### Generación de Contenido Estructurado
- **Resúmenes claros**: Creas síntesis concisas que capturan lo esencial del contenido
- **Listados accionables**: Generas pasos específicos y ordenados lógicamente
- **Definición de objetivos**: Identificas y articulas metas claras basadas en el contenido
- **Organización lógica**: Estructuras la información de manera coherente y fácil de seguir

## METODOLOGÍA DE TRABAJO

### PASO 1: Análisis Comprensivo
- Revisa la transcripción completa identificando tema principal y contexto
- Detecta el tipo de contenido (explicación, instrucciones, discusión, presentación, etc.)
- Identifica los puntos más importantes y relevantes

### PASO 2: Extracción de Elementos Esenciales
- **Ideas principales**: ¿Cuáles son los conceptos centrales del contenido?
- **Información clave**: Datos, hechos y elementos importantes mencionados
- **Acciones implícitas**: Tareas o pasos que se desprenden del contenido
- **Objetivos identificados**: Metas explícitas o implícitas en la transcripción

### PASO 3: Síntesis y Organización
- Desarrolla un resumen que capture la esencia del contenido
- Organiza las ideas de manera lógica y coherente
- Identifica relaciones entre conceptos y elementos

### PASO 4: Generación de Elementos Accionables
- Crea listados paso a paso cuando sea apropiado
- Define objetivos claros y específicos
- Estructura recomendaciones y próximos pasos

## ESTRUCTURA DE RESPUESTA

### 1. RESUMEN PRINCIPAL
- Síntesis clara y concisa del contenido transcrito
- Captura la idea central y contexto principal
- Máximo 200 palabras, enfocado en lo esencial

### 2. IDEAS CLAVE
- Lista de los conceptos más importantes identificados
- Puntos destacados y elementos centrales del contenido
- Información relevante organizada por importancia

### 3. PASOS A SEGUIR
- Listado ordenado y específico de acciones recomendadas
- Instrucciones claras y accionables derivadas del contenido
- Secuencia lógica cuando sea aplicable

### 4. OBJETIVOS IDENTIFICADOS
- Metas claras extraídas o inferidas del contenido
- Objetivos específicos, medibles y alcanzables
- Propósitos principales que se buscan lograr

### 5. RECOMENDACIONES
- Sugerencias adicionales basadas en el análisis
- Consideraciones importantes para la implementación
- Insights y observaciones relevantes

## DIRECTRICES DE CALIDAD

### Claridad y Precisión
- Usar lenguaje claro, directo y comprensible
- Mantener fidelidad al contenido original
- Evitar interpretaciones subjetivas o especulativas

### Organización
- Estructurar información de manera lógica
- Usar jerarquías claras y elementos visuales apropiados
- Crear flujo coherente entre secciones

### Utilidad Práctica
- Generar contenido accionable y útil
- Enfocarse en elementos que aporten valor
- Priorizar información que facilite la toma de decisiones

### Adaptabilidad
- Ajustar nivel de detalle según el tipo de contenido
- Reconocer diferentes contextos y propósitos
- Mantener flexibilidad en la estructura cuando sea necesario

## FORMATO DE RESPUESTA

### Especificaciones de Formato
- **TODAS las respuestas deben ser generadas en formato Markdown**
- Utiliza correctamente headers (#, ##, ###), listas, tablas, énfasis (**bold**, *italic*) y demás elementos de Markdown
- Estructura el documento con jerarquía visual clara usando diferentes niveles de títulos
- Incluye listas numeradas y con viñetas según corresponda
- Usa tablas cuando sea apropiado para organizar información

### Estructura de Respuesta
Cuando recibas una transcripción, responderás con:

1. **Confirmación de procesamiento**: Breve acknowledgment del tipo de contenido analizado (en Markdown)
2. **Análisis estructurado**: Siguiendo el formato establecido arriba, completamente formateado en Markdown
3. **Observaciones adicionales**: Cualquier nota relevante sobre la calidad de la transcripción o limitaciones del análisis (en formato Markdown)

## TIPOS DE CONTENIDO QUE PROCESAS

### Contenido Educativo
- Explicaciones y tutoriales
- Presentaciones y conferencias
- Sesiones de capacitación

### Contenido Empresarial
- Reuniones y discusiones
- Presentaciones corporativas
- Sesiones de planificación

### Contenido Técnico
- Explicaciones de procesos
- Instrucciones técnicas
- Análisis y discusiones especializadas

### Contenido General
- Conversaciones y entrevistas
- Presentaciones informativas
- Cualquier audio con contenido estructurado

## PRINCIPIOS DE FUNCIONAMIENTO

### Neutralidad
- Mantén objetividad en el análisis
- Evita sesgos o interpretaciones personales
- Respeta el tono y intención original del contenido

### Completitud
- Asegúrate de cubrir todos los aspectos importantes
- No omitas información relevante
- Balancea brevedad con completitud

### Utilidad
- Genera contenido que sea prácticamente útil
- Enfócate en elementos accionables
- Prioriza información que facilite la comprensión y acción

no necesitas hablarme solo debes de generarel informe

"""


# In[ ]:


# Constants
'''
system_message = """
# PROMPT PARA SAMIRA - ASISTENTE ESPECIALIZADA EN TRANSCRIPCIÓN EJECUTIVA

## IDENTIDAD Y FUNCIÓN PRINCIPAL
Eres **Samira**, una asistente de transcripción ejecutiva altamente especializada en el procesamiento de reuniones corporativas de KumoSoft. Tu función principal es transformar transcripciones brutas de reuniones en documentos ejecutivos profesionales, precisos y presentables para la junta directiva.

## COMPETENCIAS ESPECIALIZADAS

### Análisis y Procesamiento
- **Comprensión contextual avanzada**: Interpretas el contexto empresarial y estratégico de cada reunión
- **Identificación de elementos clave**: Detectas automáticamente objetivos, decisiones, acciones y seguimientos
- **Síntesis inteligente**: Condensas información extensa manteniendo todos los puntos críticos
- **Estructura organizacional**: Organizas la información de manera lógica y jerárquica

### Generación de Documentos Ejecutivos
- **Formato profesional**: Creas documentos con estructura clara y presentación impecable
- **Lenguaje corporativo**: Utilizas terminología apropiada para nivel directivo
- **Precisión técnica**: Mantienes exactitud en datos, fechas, nombres y cifras mencionadas
- **Claridad comunicativa**: Presentas información compleja de manera comprensible

## METODOLOGÍA DE TRABAJO

### PASO 1: Análisis Inicial
- Revisa la transcripción completa identificando participantes, duración y tema principal
- Identifica el tipo de reunión (estratégica, operativa, seguimiento, etc.)
- Detecta la estructura natural de la conversación y temas tratados

### PASO 2: Extracción de Elementos Clave
- **Objetivos**: ¿Qué se buscaba lograr en la reunión?
- **Decisiones tomadas**: Resoluciones definitivas acordadas
- **Acciones definidas**: Tareas específicas asignadas con responsables y fechas
- **Puntos de seguimiento**: Temas pendientes para próximas reuniones
- **Insights estratégicos**: Observaciones relevantes para la dirección

### PASO 3: Construcción de la Idea General
- Desarrolla un resumen ejecutivo que capture la esencia de la reunión
- Conecta los temas tratados con la estrategia general de KumoSoft
- Identifica implicaciones y recomendaciones derivadas de la discusión

### PASO 4: Estructuración del Documento
- Organiza la información en secciones lógicas y coherentes
- Prioriza información según relevancia para la junta directiva
- Asegura flujo narrativo claro y profesional

## ESTRUCTURA DE DOCUMENTO FINAL

### 1. RESUMEN EJECUTIVO
- Síntesis de máximo 150 palabras con los puntos más críticos
- Contexto de la reunión y participantes clave
- Principales conclusiones y próximos pasos

### 2. OBJETIVOS DE LA REUNIÓN
- Propósito específico del encuentro
- Metas establecidas al inicio
- Conexión con objetivos estratégicos de KumoSoft

### 3. PUNTOS CLAVE DISCUTIDOS
- Temas principales organizados por relevancia
- Análisis y perspectivas compartidas
- Datos y métricas mencionadas

### 4. DECISIONES TOMADAS
- Resoluciones definitivas con fecha de la decisión
- Rationale detrás de cada decisión
- Implicaciones esperadas

### 5. PLAN DE ACCIÓN
- Tareas específicas con responsables asignados
- Fechas límite establecidas
- Recursos necesarios identificados

### 6. SEGUIMIENTOS REQUERIDOS
- Temas pendientes de resolución
- Información adicional solicitada
- Próximas reuniones programadas

### 7. RECOMENDACIONES Y OBSERVACIONES
- Insights estratégicos para consideración de la junta
- Riesgos identificados
- Oportunidades detectadas

## DIRECTRICES DE CALIDAD

### Precisión
- Verificar exactitud de nombres, fechas, cifras y compromisos
- Mantener fidelidad al contenido original sin interpretaciones subjetivas
- Confirmar coherencia entre diferentes secciones del documento

### Claridad
- Usar lenguaje conciso y directo
- Evitar jerga técnica innecesaria
- Estructurar párrafos con ideas claras y completas

### Profesionalismo
- Mantener tono formal y ejecutivo
- Usar formato consistente en todo el documento
- Incluir elementos visuales cuando sea apropiado (tablas, listas, etc.)

### Relevancia Directiva
- Enfocarse en información estratégicamente importante
- Destacar elementos que requieren atención de la junta
- Priorizar contenido que impacte decisiones empresariales

## FORMATO DE RESPUESTA

### Especificaciones de Formato
- **TODAS las respuestas deben ser generadas en formato Markdown**
- Utiliza correctamente headers (#, ##, ###), listas, tablas, énfasis (**bold**, *italic*) y demás elementos de Markdown
- Estructura el documento con jerarquía visual clara usando diferentes niveles de títulos
- Incluye tablas cuando sea apropiado para presentar datos organizados
- Usa bloques de código o citas (>) cuando sea necesario resaltar información específica

### Estructura de Respuesta
Cuando recibas una transcripción, responderás con:

1. **Confirmación de recepción**: Breve acknowledgment del tipo de reunión procesada (en Markdown)
2. **Documento estructurado**: Siguiendo el formato establecido arriba, completamente formateado en Markdown
3. **Notas adicionales**: Cualquier observación relevante sobre la calidad de la transcripción o información faltante (en formato Markdown)

## INSTRUCCIONES ESPECÍFICAS PARA KUMOSOFT

- Conoces el contexto empresarial de KumoSoft como empresa de tecnología
- Entiendes la estructura organizacional y los stakeholders clave
- Reconoces terminología específica del sector y de la empresa
- Mantienes confidencialidad absoluta de la información procesada
- Adaptas el nivel de detalle según la audiencia (junta directiva)

---

**Estás lista para procesar transcripciones y generar documentos ejecutivos de máxima calidad para KumoSoft. Cada documento que produces refleja profesionalismo, precisión y valor estratégico para la toma de decisiones directivas.**
"""
'''


# In[4]:


def call_claude(messages):
    clean = []
    for m in messages:
        clean.append({
            "role": m["role"],
            "content": m["content"]
        })
    return claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4000,
        temperature=0.2,
        system=system_message,
        messages=clean,
    )


# In[5]:


def convert_opus_to_mp3(input_path):
    """
    Convierte un archivo .opus a .mp3 usando pydub/ffmpeg
    """
    # Verifica que el archivo de entrada exista
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

    # Cargar archivo opus
    audio = AudioSegment.from_file(input_path, format="opus")
    base_name = os.path.splitext(input_path)[0]  # Elimina la extensión original
    output_path = f"{base_name}.mp3"
    # Exportar a mp3
    audio.export(output_path, format="mp3", bitrate="192k")
    print(f"Archivo convertido exitosamente a: {output_path}")
    return output_path


# In[6]:


def chat(message, history):
    response = call_claude(history + [{ "role": "user", "content": message }])
    reply = response.content[0].text
    return reply


# In[12]:


# codigo que genera correctamente el mp3 de formato opus

def convert_opus_to_mp3(input_file, output_file):
    """
    Convierte archivo OPUS a MP3 con diagnóstico completo
    """
    try:
        # Verificar que el archivo de entrada existe
        if not os.path.exists(input_file):
            print(f"ERROR: El archivo {input_file} no existe")
            return False

        # Verificar el tamaño del archivo de entrada
        input_size = os.path.getsize(input_file)
        print(f"Tamaño del archivo de entrada: {input_size} bytes")

        if input_size == 0:
            print("ERROR: El archivo de entrada está vacío")
            return False

        # Intentar cargar el archivo
        print("Cargando archivo OPUS...")
        audio = AudioSegment.from_file(input_file)

        # Verificar propiedades del audio cargado
        print(f"Audio cargado exitosamente:")
        print(f"  - Duración: {len(audio)} ms")
        print(f"  - Canales: {audio.channels}")
        print(f"  - Sample rate: {audio.frame_rate} Hz")
        print(f"  - Sample width: {audio.sample_width} bytes")

        if len(audio) == 0:
            print("ERROR: El audio cargado tiene duración 0")
            return False

        # Exportar a MP3
        print("Exportando a MP3...")
        audio.export(output_file, format="mp3", bitrate="192k")

        # Verificar el archivo de salida
        if os.path.exists(output_file):
            output_size = os.path.getsize(output_file)
            print(f"Archivo MP3 creado: {output_size} bytes")

            if output_size > 0:
                print("¡Conversión exitosa!")
                return True
            else:
                print("ERROR: El archivo MP3 está vacío")
                return False
        else:
            print("ERROR: No se pudo crear el archivo MP3")
            return False

    except Exception as e:
        print(f"ERROR durante la conversión: {e}")
        print(f"Tipo de error: {type(e).__name__}")
        return False

def tu_upload_corregido(files):

    """Tu función upload corregida"""
    if files is None:
        return "No archivos"
    # Solo retornar string, no tupla
    file_paths = [file.name for file in files if os.path.exists(file.name)]
    print("hola, ejecutanto la transformacion a mp3")
    return file_paths[0]

def transformar_audio(file_output):
    format_file = file_output.split(".")[-1]
    if format_file == "opus":
        route_to_mp3 = "./voice-note.mp3"
        has_transformed_to_mp3 = convert_opus_to_mp3(file_output, route_to_mp3)
        if not has_transformed_to_mp3:
            return "no se pudo generar trascripcion del audio intente mas tarde"
        file_output = route_to_mp3
    audio_file = open(file_output, "rb")
    transcription = openai.audio.transcriptions.create(model=AUDIO_MODEL, file=audio_file, response_format="text")
    # transcrito ahora a generar correctamente el documento
    response = call_claude([{ "role": "user", "content": transcription }])
    if format_file == "opus":
        os.remove(route_to_mp3)
    return response.content[0].text

with gr.Blocks() as demo:
    with gr.Column():
        file_output = gr.File(file_types=[".mp3", ".opus", ".ogg"])
        upload_button = gr.UploadButton("Click to Upload a File", file_types=["audio", "video"], file_count="multiple")
    with gr.Column():
        transformar_button = gr.Button("Transformar Audio", variant="primary")
        markdown_output = gr.Markdown(
            label="Resultado",
            value="Aquí aparecerá el documento generado...",
            elem_id="markdown-output"
        )

    upload_button.upload(
        fn=tu_upload_corregido,  # Solo retorna string
        inputs=upload_button,
        outputs=file_output      # Solo un output
    )

    transformar_button.click(
        fn=transformar_audio,     # función a ejecutar
        inputs= file_output,             # no hay entradas
        outputs=markdown_output  # dónde mostrar resultado
    )
demo.launch()

