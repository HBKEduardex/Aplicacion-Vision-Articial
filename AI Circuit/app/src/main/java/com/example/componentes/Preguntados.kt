package com.example.componentes

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.example.componentes.databinding.ActivityPreguntadosBinding

class Preguntados : AppCompatActivity() {

    private lateinit var binding: ActivityPreguntadosBinding
    private lateinit var prediccionReal: String

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityPreguntadosBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Corregido: usar la misma clave que usaste al enviar desde MainActivity
        prediccionReal = intent.getStringExtra("prediccion_detalle") ?: ""

        // Mostrar la pregunta
        binding.textView.text = getString(R.string.pregunta)

        // Configurar los botones
        val opciones = mapOf(
            binding.resistencia to "Resistor",
            binding.cap to "Capacitor",
            binding.motor to "Motor",
            binding.trans to "Transistor",
            binding.rasp to "Raspberry",
            binding.arduino to "Arduino",
            binding.tiva to "Tiva",
            binding.stm to "STM32",
            binding.esp to "ESP32"
        )

        for ((boton, valor) in opciones) {
            boton.setOnClickListener {
                val intent = Intent(this, ResultadoActivity::class.java).apply {
                    putExtra("SELECCION", valor)
                    putExtra("PREDICCION", prediccionReal)
                }
                startActivity(intent)
                finish()
            }
        }
    }
}
