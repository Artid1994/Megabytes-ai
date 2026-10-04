package com.example.childbrainsnn

import android.Manifest
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import kotlinx.coroutines.delay
import java.util.Locale

/**
 * MainActivity — Entry point for the Child Brain SNN Engine.
 *
 * Lifecycle:
 *   ON_RESUME  -> start SNN simulation thread + CameraX
 *   ON_PAUSE   -> stop simulation thread
 *   ON_DESTROY -> shut down TTS + controller
 *
 * Permissions: CAMERA (for CameraX visual diff), RECORD_AUDIO (declared per
 * design.json; engine uses simulated mic by default).
 */
class MainActivity : ComponentActivity() {

    private var tts: TextToSpeech? = null
    private var engine: ChildBrainSNN? = null
    private var analyzer: VisualDiffAnalyzer? = null
    private var controller: SNNController? = null
    private var hasCameraPermission: Boolean = false

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        hasCameraPermission = permissions[Manifest.permission.CAMERA] ?: false
        if (!hasCameraPermission) {
            Toast.makeText(
                this,
                "Camera permission required. SNN will use simulated visual input.",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // -- Request permissions --------------------------------------------
        permissionLauncher.launch(
            arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
        )

        // -- Initialize TextToSpeech ---------------------------------------
        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = Locale.getDefault()
                tts?.setSpeechRate(0.9f)
                Log.d("MainActivity", "TTS initialized, lang = ${Locale.getDefault()}")
            } else {
                Log.e("MainActivity", "TTS init failed")
            }
        }

        // -- Initialize SNN engine -----------------------------------------
        engine = ChildBrainSNN()
        analyzer = VisualDiffAnalyzer()
        controller = SNNController(engine!!, tts)

        // -- Compose content ------------------------------------------------
        setContent {
            ChildBrainApp(
                engine = engine!!,
                controller = controller!!,
                analyzer = analyzer!!,
                hasCameraPermission = hasCameraPermission
            )
        }
    }

    override fun onResume() {
        super.onResume()
        if (hasCameraPermission) {
            analyzer?.reset()
        }
        controller?.startSimulation()
    }

    override fun onPause() {
        super.onPause()
        controller?.stopSimulation()
    }

    override fun onDestroy() {
        super.onDestroy()
        controller?.stopSimulation()
        tts?.shutdown()
        engine = null
        analyzer = null
        controller = null
    }
}

// ===========================================================================
// Composable: Root app screen
// ===========================================================================

@Composable
fun ChildBrainApp(
    engine: ChildBrainSNN,
    controller: SNNController,
    analyzer: VisualDiffAnalyzer,
    hasCameraPermission: Boolean
) {
    val lifecycleOwner = LocalLifecycleOwner.current

    // Start simulation automatically when the screen appears
    DisposableEffect(Unit) {
        controller.startSimulation()
        onDispose { controller.stopSimulation() }
    }

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = Color(0xFF0A0A1A)
    ) {
        Box(modifier = Modifier.fillMaxSize()) {

            // -- CameraX live preview (feeds visualInput into the engine) --
            if (hasCameraPermission) {
                CameraPreviewWithAnalysis(
                    lifecycleOwner = lifecycleOwner,
                    engine = engine,
                    analyzer = analyzer,
                    modifier = Modifier.fillMaxSize()
                )
            } else {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(Color(0xFF000000)),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "Camera permission pending — SNN running in simulation mode",
                        color = Color(0xFF888888),
                        fontSize = 14.sp
                    )
                }
            }

            // -- Neural network visualizer overlay ---------------------------
            NeuralVisualizerCanvas(
                engine = engine,
                modifier = Modifier
                    .fillMaxSize()
                    .padding(8.dp)
            )

            // -- Stats overlay (top-left) ------------------------------------
            StatsOverlay(
                engine = engine,
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(8.dp)
            )

            // -- Start/Stop button (bottom-right) ---------------------------
            var started by remember { mutableStateOf(false) }
            Card(
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .padding(16.dp),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0x33FFFFFF))
            ) {
                Box(
                    modifier = Modifier
                        .background(
                            brush = Brush.horizontalGradient(
                                colors = listOf(Color(0xFF4A90D9), Color(0xFF50E3C2))
                            )
                        )
                        .padding(horizontal = 20.dp, vertical = 12.dp)
                ) {
                    Text(
                        text = if (started) "RUNNING" else "START SNN",
                        color = Color.White,
                        fontWeight = FontWeight.Bold,
                        fontSize = 14.sp
                    )
                }
            }
        }
    }
}
