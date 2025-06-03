package com.example.componentes

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class ResultadoActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_resultado)

        val textoRespuesta = findViewById<TextView>(R.id.textView2)
        val imagenReferencia = findViewById<ImageView>(R.id.imageView3)
        val aviso = findViewById<TextView>(R.id.textView4)
        val volver = findViewById<Button>(R.id.volver)

        val prediccion = intent.getStringExtra("PREDICCION") ?: ""
        val eleccion = intent.getStringExtra("SELECCION") ?: ""

        val acerto = prediccion.equals(eleccion, ignoreCase = true)

        textoRespuesta.text = if (acerto) {
            "¡Lo hiciste bien! La respuesta coincide con el modelo."
        } else {
            "No coincidió 😔 El modelo predijo: $prediccion"
        }

        val imagenId = when (prediccion.lowercase()) {
            "arduino" -> R.drawable.arduino
            "tiva" -> R.drawable.tiva
            "raspberry" -> R.drawable.raspberry  // corrige el nombre si es typo
            "stm32" -> R.drawable.stm32
            "esp32" -> R.drawable.esp32
            "resistor" -> R.drawable.res
            "motor" -> R.drawable.motor
            "capacitor" -> R.drawable.cap
            "transistor" -> R.drawable.trans
            else -> R.drawable.bgapp
        }

        imagenReferencia.setImageResource(imagenId)

        aviso.text = getString(R.string.Aviso)

        volver.setOnClickListener {
            val intent = Intent(this, MainActivity::class.java)
            intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_NEW_TASK
            startActivity(intent)
            finish()
        }
    }
}
