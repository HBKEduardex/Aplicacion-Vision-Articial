# 📱 Componentes - App con IA y Cámara

Esta es una aplicación Android desarrollada en Kotlin que permite **identificar componentes electrónicos** (como microcontroladores y pasivos) a partir de una foto tomada con la cámara del dispositivo. Utiliza modelos de **TensorFlow Lite** para la clasificación y ofrece una experiencia interactiva con retroalimentación visual y textual.

## 🚀 Funcionalidades

- 📸 Captura de imagen en tiempo real con CameraX  
- 🧠 Clasificación automática usando modelos `.tflite`:
  - Clasificación binaria: Microcontrolador vs Componente
  - Identificación del tipo específico
- ❓ Pantalla interactiva: el usuario intenta adivinar el componente
- ✅ Resultado final con verificación, imagen de referencia y aviso
- 📱 Interfaz responsiva con orientación vertical fija

## 🧠 Modelos de IA

Se utilizan tres modelos de **TensorFlow Lite**, incluidos en la carpeta `assets/`:

| Modelo             | Tipo                    | Objetivo                                 |
|--------------------|-------------------------|-------------------------------------------|
| `first.tflite`     | Binario                 | ¿Es microcontrolador o componente?        |
| `micro.tflite`     | Multiclase              | `Arduino`, `ESP32`, `STM32`, `Tiva`, `Raspberry` |
| `componentes.tflite` | Multiclase            | `Resistor`, `Motor`, `Capacitor`, `Transistor` |

### 📁 Estructura del proyecto

```plaintext
app/
├── src/
│   └── main/
│       ├── java/com/example/componentes/
│       │   ├── MainActivity.kt
│       │   ├── Preguntados.kt
│       │   └── ResultadoActivity.kt
│       ├── res/
│       │   ├── layout/
│       │   │   ├── activity_main.xml
│       │   │   ├── activity_preguntados.xml
│       │   │   └── activity_resultado.xml
│       │   ├── drawable/
│       │   │   ├── arduino.jpeg
│       │   │   ├── cap.jpg
│       │   │   ├── esp32.jpeg
│       │   │   ├── motor.jpeg
│       │   │   ├── raspberry.jpeg
│       │   │   ├── res.jpg
│       │   │   ├── stm32.jpeg
│       │   │   ├── tiva.jpeg
│       │   │   └── trans.jpg
│       │   └── values/
│       │       └── strings.xml
│       └── assets/
│           ├── first.tflite
│           ├── micro.tflite
│           └── componentes.tflite

```

## 🧩 Dependencias principales

- [CameraX](https://developer.android.com/training/camerax) para captura de imagen
- [TensorFlow Lite](https://www.tensorflow.org/lite) para inferencia
- ViewBinding y ConstraintLayout

## 🔒 Permisos necesarios

```xml
<uses-permission android:name="android.permission.CAMERA" />
```

📸 Cómo funciona
El usuario inicia la app y presiona "Detectar".

Se toma una foto con la cámara.

Se clasifica automáticamente y se pasa a la pantalla de preguntas.

El usuario intenta adivinar el componente mostrado.

Se muestra el resultado final, indicando si acertó o no, junto a una imagen de referencia.

📱 Compatibilidad
Funciona en dispositivos Android con API 21 (Android 5.0) en adelante.

Probado en dispositivo real con Android 14.

⚙️ Recomendaciones
Asegúrate de tener los modelos .tflite dentro de la carpeta assets/.

Las imágenes de referencia deben estar en res/drawable/ con los siguientes nombres:
arduino.jpeg, tiva.jpeg, raspberry.jpeg, stm32.jpeg, esp32.jpeg
res.jpg, motor.jpeg, cap.jpg, trans.jpg

👨‍💻 Autores
Desarrollado por:
Adrián Eduardo Vargas LLanquipacha 
, Israel Silva Bernal
, Belén Medina
, Hector Fernández
Proyecto académico para la identificación de componentes electrónicos mediante visión por computadora y redes neuronales.
