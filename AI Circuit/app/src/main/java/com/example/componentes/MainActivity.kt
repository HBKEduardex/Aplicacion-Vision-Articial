package com.example.componentes

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.componentes.databinding.ActivityMainBinding
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val cameraPermissionCode = 100
    private lateinit var imageCapture: ImageCapture
    private lateinit var cameraExecutor: ExecutorService

    private lateinit var tfliteBinario: Interpreter
    private lateinit var tfliteMicro: Interpreter
    private lateinit var tfliteComponente: Interpreter

    private var modelosListos = false

    private val microLabels = listOf("Arduino", "ESP32", "Raspberry", "STM32", "Tiva")
    private val componenteLabels = listOf("Capacitor", "Motor", "Resistor", "Transistor")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()

        Thread {
            tfliteBinario = Interpreter(loadModelFile("first.tflite"))
            tfliteMicro = Interpreter(loadModelFile("micro.tflite"))
            tfliteComponente = Interpreter(loadModelFile("componentes.tflite"))
            modelosListos = true
        }.start()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), cameraPermissionCode)
        }

        binding.btnDetectar.setOnClickListener {
            if (modelosListos) {
                takePhoto()
            } else {
                Toast.makeText(this, "Cargando modelos...", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun loadModelFile(fileName: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(fileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            imageCapture = ImageCapture.Builder().build()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture)
            } catch (e: Exception) {
                Log.e("CameraX", "Error al iniciar cámara", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto() {
        val photoFile = File.createTempFile("captura_", ".jpg", cacheDir)
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        imageCapture.takePicture(outputOptions, ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    val bitmap = BitmapFactory.decodeFile(photoFile.absolutePath)
                    classifyGeneral(bitmap)
                }

                override fun onError(exception: ImageCaptureException) {
                    Toast.makeText(applicationContext, "Error al capturar imagen", Toast.LENGTH_SHORT).show()
                    Log.e("CameraX", "Error al capturar imagen", exception)
                }
            }
        )
    }

    private fun classifyGeneral(bitmap: Bitmap) {
        val resized224 = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val inputBufferBinario = convertBitmapToByteBufferRGB224(resized224)

        val output = Array(1) { FloatArray(1) }
        tfliteBinario.run(inputBufferBinario, output)

        val probMicro = output[0][0]
        val probComp = 1f - probMicro

        val esComponente = probComp >= 0.95f
        val etiquetaPrincipal = if (esComponente) "Componente" else "Microcontrolador"

        var subLabel = ""
        try {
            subLabel = if (esComponente) {
                classifyComponentModel(bitmap).first
            } else {
                classifyMicroModel(resized224).first
            }
        } catch (e: Exception) {
            subLabel = "Error: ${e.message}"
        }

        val intent = Intent(this, Preguntados::class.java).apply {
            putExtra("prediccion_general", etiquetaPrincipal)
            putExtra("prediccion_detalle", subLabel)
        }
        startActivity(intent)
    }

    private fun classifyComponentModel(bitmap: Bitmap): Pair<String, String> {
        val resized128 = Bitmap.createScaledBitmap(bitmap, 128, 128, true)
        val inputBuffer = convertBitmapToByteBufferRGB128(resized128)
        val output = Array(1) { FloatArray(componenteLabels.size) }
        tfliteComponente.run(inputBuffer, output)
        return interpretOutput(output[0], componenteLabels)
    }

    private fun classifyMicroModel(resized224: Bitmap): Pair<String, String> {
        val inputBuffer = convertBitmapToByteBufferRGB224(resized224)
        val output = Array(1) { FloatArray(microLabels.size) }
        tfliteMicro.run(inputBuffer, output)
        return interpretOutput(output[0], microLabels)
    }

    private fun interpretOutput(scores: FloatArray, labels: List<String>): Pair<String, String> {
        val idx = scores.indices.maxByOrNull { scores[it] } ?: -1
        val label = labels.getOrElse(idx) { "Desconocido" }
        val detalles = scores.mapIndexed { i, prob ->
            "%s: %.4f".format(labels.getOrElse(i) { "Clase $i" }, prob)
        }.joinToString("\n")
        return Pair(label, detalles)
    }

    private fun convertBitmapToByteBufferRGB224(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3)
        byteBuffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(224 * 224)
        bitmap.getPixels(pixels, 0, 224, 0, 0, 224, 224)
        for (pixel in pixels) {
            val r = (pixel shr 16 and 0xFF) / 255.0f
            val g = (pixel shr 8 and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            byteBuffer.putFloat(r)
            byteBuffer.putFloat(g)
            byteBuffer.putFloat(b)
        }
        return byteBuffer
    }

    private fun convertBitmapToByteBufferRGB128(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * 128 * 128 * 3)
        byteBuffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(128 * 128)
        bitmap.getPixels(pixels, 0, 128, 0, 0, 128, 128)
        for (pixel in pixels) {
            val r = (pixel shr 16 and 0xFF) / 255.0f
            val g = (pixel shr 8 and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            byteBuffer.putFloat(r)
            byteBuffer.putFloat(g)
            byteBuffer.putFloat(b)
        }
        return byteBuffer
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        if (::tfliteBinario.isInitialized) tfliteBinario.close()
        if (::tfliteMicro.isInitialized) tfliteMicro.close()
        if (::tfliteComponente.isInitialized) tfliteComponente.close()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == cameraPermissionCode && grantResults.isNotEmpty() &&
            grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            Toast.makeText(this, "Permiso de cámara denegado", Toast.LENGTH_SHORT).show()
        }
    }
}
