@file:Suppress("unused")

package com.example.childbrainsnn

import android.speech.tts.TextToSpeech
import android.util.Log
import androidx.annotation.OptIn
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
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
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.Stable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.delay
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.min
import kotlin.math.sin

// ===========================================================================
// UI Layer Constants
// ===========================================================================

internal val layerColors = listOf(
    Color(0xFF4CAF50),  // Visual — green
    Color(0xFF2196F3),  // Auditory — blue
    Color(0xFFF44336),  // Hidden — red
    Color(0xFFFFEB3B),  // Motor — yellow
)

internal val layerNames = listOf("Visual", "Auditory", "Hidden", "Motor")

// ===========================================================================
// Section 1: Zero-Allocation RingBuffer<Int> (capacity 512, preallocated)
// Matches design.json: "RingBuffer<Int>, capacity 512, preallocate: true"
// ===========================================================================

@Stable
class SpikeRingBuffer(private val cap: Int) {
    private val buf: IntArray = IntArray(cap)
    @JvmField var tail: Int = 0
    @JvmField var cnt: Int = 0

    fun push(neuronId: Int) {
        if (cnt < cap) {
            buf[(tail + cnt) % cap] = neuronId
            cnt++
        }
    }

    fun pop(): Int {
        if (cnt == 0) return -1
        val v = buf[tail]
        tail = (tail + 1) % cap
        cnt--
        return v
    }

    fun clear() {
        tail = 0
        cnt = 0
    }

    fun size(): Int = cnt
    fun capacity(): Int = cap

    fun snapshotInto(out: IntArray): Int {
        val toCopy = if (cnt < out.size) cnt else out.size
        for (i in 0 until toCopy) {
            out[i] = buf[(tail + i) % cap]
        }
        return toCopy
    }
}

// ===========================================================================
// Section 2: ChildBrainSNN Engine — Single File, Zero-Allocation
// LIF Neuron + STDP Synapse + CameraX visual diff + simulated mic +
// Compose Visualizer Canvas + TextToSpeech motor output
// ===========================================================================

@Stable
class ChildBrainSNN {
    // -- Layer sizes (from design.json) --------------------------------------
    @JvmField val VISUAL_SIZE: Int = 256
    @JvmField val AUDITORY_SIZE: Int = 128
    @JvmField val HIDDEN_SIZE: Int = 64
    @JvmField val MOTOR_SIZE: Int = 32
    private val TOTAL: Int = VISUAL_SIZE + AUDITORY_SIZE + HIDDEN_SIZE + MOTOR_SIZE  // 480

    // -- Layer offsets (flat address space) ----------------------------------
    @JvmField val VISUAL_START: Int = 0          // 0..255
    @JvmField val AUDITORY_START: Int = VISUAL_SIZE          // 256..383
    @JvmField val HIDDEN_START: Int = AUDITORY_START + AUDITORY_SIZE  // 384..447
    @JvmField val MOTOR_START: Int = HIDDEN_START + HIDDEN_SIZE       // 448..479

    // -- LIF parameters (design.json) ----------------------------------------
    private val V_REST = 0.0f
    private val V_THRESHOLD = 1.0f
    private val TAU_M_MS = 20.0f      // membrane time constant
    private val TAU_REF_MS = 5.0f     // refractory period

    // -- STDP parameters (design.json) ---------------------------------------
    private val TAU_PLUS = 20.0f      // pre-synaptic trace decay
    private val TAU_MINUS = 40.0f     // post-synaptic trace decay
    private val A_PLUS = 0.005f       // LTP amplitude
    private val A_MINUS = -0.008f     // LTD amplitude
    private val W_MIN = 0.0f
    private val W_MAX = 1.0f

    // -- Exponential synapse decay -------------------------------------------
    private val TAU_SYN = 5.0f

    // -- Neuron state arrays (preallocated, zero-allocation in step loop) ----
    @JvmField val membrane: FloatArray = FloatArray(TOTAL)
    @JvmField val refractory: FloatArray = FloatArray(TOTAL)
    @JvmField val neuronSpiked: BooleanArray = BooleanArray(TOTAL)

    // -- Per-neuron accumulated input current (reset each step) -------------
    @JvmField val inputCurrent: FloatArray = FloatArray(TOTAL)

    // -- Spike event ring buffer (design.json: capacity 512) ---------------
    @JvmField val spikeBuffer: SpikeRingBuffer = SpikeRingBuffer(512)

    // -- Synapse connectivity (generated at init, read-only after init) -----
    @JvmField val connCount: Int
    @JvmField val preIdx: IntArray
    @JvmField val postIdx: IntArray

    // -- Synapse state arrays (preallocated) -------------------------------
    @JvmField val weights: FloatArray
    @JvmField val synCurrent: FloatArray
    @JvmField val preTrace: FloatArray
    @JvmField val postTrace: FloatArray

    // -- STDP: last spike time per neuron ------------------------------------
    @JvmField val lastSpikeTime: FloatArray = FloatArray(TOTAL) { -10000f }

    // -- External input buffers ----------------------------------------------
    @JvmField val visualInput: FloatArray = FloatArray(VISUAL_SIZE)
    @JvmField val auditoryInput: FloatArray = FloatArray(AUDITORY_SIZE)

    // -- Motor output buffer -----------------------------------------------
    @JvmField val motorOutput: IntArray = IntArray(MOTOR_SIZE)

    // -- Simulation time ----------------------------------------------------
    @JvmField var simTime: Float = 0.0f

    // -- Recent spike history for visualizer (ring buffer) ------------------
    @JvmField val recentSpikes: IntArray = IntArray(256)
    @JvmField var recentSpikeWrite: Int = 0
    @JvmField var recentSpikeCount: Int = 0
    private val spikeSnapshot: IntArray = IntArray(256)

    // -- PRNG state for zero-allocation noise (XOR-shift) -------------------
    private var noiseState: Long = 0x123456789ABCDEFL

    init {
        // Generate fully-connected layers (skip self-connections for recurrent)
        // Total synapses: 256*64 + 128*64 + 64*63 + 64*32 = 30560
        val v2h = VISUAL_SIZE * HIDDEN_SIZE     // 16384
        val a2h = AUDITORY_SIZE * HIDDEN_SIZE   // 8192
        val h2h = HIDDEN_SIZE * (HIDDEN_SIZE - 1) // 4032 (skip self)
        val h2m = HIDDEN_SIZE * MOTOR_SIZE      // 2048

        connCount = v2h + a2h + h2h + h2m
        preIdx = IntArray(connCount)
        postIdx = IntArray(connCount)
        weights = FloatArray(connCount)
        synCurrent = FloatArray(connCount)
        preTrace = FloatArray(connCount)
        postTrace = FloatArray(connCount)

        var offset = 0
        // visual -> hidden
        offset = fillConnections(offset, VISUAL_START, VISUAL_SIZE, HIDDEN_START, HIDDEN_SIZE)
        // auditory -> hidden
        offset = fillConnections(offset, AUDITORY_START, AUDITORY_SIZE, HIDDEN_START, HIDDEN_SIZE)
        // hidden -> hidden (recurrent, skip self)
        for (p in 0 until HIDDEN_SIZE) {
            for (q in 0 until HIDDEN_SIZE) {
                if (p == q) continue
                preIdx[offset] = HIDDEN_START + p
                postIdx[offset] = HIDDEN_START + q
                noiseState = noiseState xor (noiseState shl 13)
                noiseState = noiseState xor (noiseState ushr 7)
                noiseState = noiseState xor (noiseState shl 17)
                weights[offset] = 0.3f + 0.5f * ((noiseState ushr 33) and 0xFF) / 255f
                offset++
            }
        }
        // hidden -> motor
        offset = fillConnections(offset, HIDDEN_START, HIDDEN_SIZE, MOTOR_START, MOTOR_SIZE)

        Log.d("ChildBrainSNN", "Initialized: ${TOTAL} neurons, ${connCount} synapses")
    }

    private fun fillConnections(
        offset: Int, preStart: Int, preSize: Int,
        postStart: Int, postSize: Int
    ): Int {
        var idx = offset
        for (p in 0 until preSize) {
            for (q in 0 until postSize) {
                preIdx[idx] = preStart + p
                postIdx[idx] = postStart + q
                noiseState = noiseState xor (noiseState shl 13)
                noiseState = noiseState xor (noiseState ushr 7)
                noiseState = noiseState xor (noiseState shl 17)
                weights[idx] = 0.3f + 0.5f * ((noiseState ushr 33) and 0xFF) / 255f
                idx++
            }
        }
        return idx
    }

    /** Zero-allocation PRNG: advances internal state, returns float in [0,1) */
    private fun nextRand(): Float {
        noiseState = noiseState xor (noiseState shl 13)
        noiseState = noiseState xor (noiseState ushr 7)
        noiseState = noiseState xor (noiseState shl 17)
        return ((noiseState ushr 33) and 0xFF).toFloat() / 255f
    }

    // -- External input setters ----------------------------------------------

    /** Set visual spike-rate input (256 values in [0,1]) from CameraX diff. */
    fun setVisualInput(input: FloatArray, size: Int) {
        val n = minOf(size, VISUAL_SIZE)
        for (i in 0 until n) {
            visualInput[i] = input[i].coerceIn(0f, 1f)
        }
        for (i in n until VISUAL_SIZE) visualInput[i] = 0f
    }

    /** Simulate microphone input: frequency-encoded spike rates for 128 neurons. */
    fun simulateAudio() {
        val t = simTime * 0.001f
        val base = 4.0f + 2.0f * sin(t * 0.5f)
        for (i in 0 until AUDITORY_SIZE) {
            val freq = base + i * 0.15f
            val envelope = (sin(t * freq * 0.002f) * 0.5f + 0.5f)
            val noise = nextRand() * 0.1f
            auditoryInput[i] = (envelope + noise).coerceIn(0f, 1f)
        }
    }

    /** Reset the audio simulator state (for fresh start). */
    fun resetAudio() {
        for (i in 0 until AUDITORY_SIZE) auditoryInput[i] = 0f
    }

    // -- Main simulation step (zero-allocation hot path) --------------------

    fun step(dt: Float) {
        simTime += dt

        val decaySyn = exp(-dt / TAU_SYN)
        val decayMem = exp(-dt / TAU_M_MS)
        val decayPre = exp(-dt / TAU_PLUS)
        val decayPost = exp(-dt / TAU_MINUS)

        // 1. Decay synaptic currents, STDP traces
        for (i in 0 until connCount) {
            synCurrent[i] *= decaySyn
            preTrace[i] *= decayPre
            postTrace[i] *= decayPost
        }

        // 2. Clear per-neuron input current
        for (n in 0 until TOTAL) {
            inputCurrent[n] = 0f
        }

        // 3. Propagate spikes from PREVIOUS step into synaptic currents
        for (i in 0 until connCount) {
            if (neuronSpiked[preIdx[i]]) {
                synCurrent[i] += weights[i]
            }
        }

        // 3b. Sum incoming synaptic currents per postsynaptic neuron
        for (i in 0 until connCount) {
            inputCurrent[postIdx[i]] += synCurrent[i]
        }

        // 4. Inject external sensory input (rate encoding -> current)
        for (i in 0 until VISUAL_SIZE) {
            val v = visualInput[i]
            if (v > 0.01f) inputCurrent[i] += v
        }
        for (i in 0 until AUDITORY_SIZE) {
            val v = auditoryInput[i]
            if (v > 0.01f) inputCurrent[AUDITORY_START + i] += v
        }

        // 5. Clear spike buffer for this step
        spikeBuffer.clear()

        // 6. Clear previous spike flags
        for (n in 0 until TOTAL) {
            neuronSpiked[n] = false
        }

        // 7. LIF neuron dynamics
        for (n in 0 until TOTAL) {
            if (refractory[n] > 0f) {
                refractory[n] -= dt
                membrane[n] = membrane[n] * decayMem
                if (membrane[n] < V_REST) membrane[n] = V_REST
            } else {
                membrane[n] = membrane[n] * decayMem + inputCurrent[n] * dt * 0.15f
                if (membrane[n] < V_REST) membrane[n] = V_REST
            }
        }

        // 8. Threshold check -> spikes
        for (n in 0 until TOTAL) {
            if (refractory[n] <= 0f && membrane[n] >= V_THRESHOLD) {
                membrane[n] = V_REST
                refractory[n] = TAU_REF_MS
                neuronSpiked[n] = true
                spikeBuffer.push(n)
                lastSpikeTime[n] = simTime
                recentSpikes[recentSpikeWrite] = n
                recentSpikeWrite = (recentSpikeWrite + 1) % recentSpikes.size
                if (recentSpikeCount < recentSpikes.size) recentSpikeCount++
            }
        }

        // 9. STDP weight updates
        for (i in 0 until connCount) {
            val pre = preIdx[i]
            val post = postIdx[i]
            if (neuronSpiked[post]) {
                weights[i] += A_PLUS * preTrace[i]
                postTrace[i] += 1.0f
            }
            if (neuronSpiked[pre]) {
                weights[i] += A_MINUS * postTrace[i]
                preTrace[i] += 1.0f
            }
            if (weights[i] < W_MIN) weights[i] = W_MIN
            if (weights[i] > W_MAX) weights[i] = W_MAX
        }

        // 10. Motor output
        for (i in 0 until MOTOR_SIZE) {
            motorOutput[i] = if (neuronSpiked[MOTOR_START + i]) 1 else 0
        }
    }

    // -- Query methods for visualization ------------------------------------

    fun snapshot(): FloatArray {
        val snap = FloatArray(TOTAL)
        for (i in 0 until TOTAL) snap[i] = membrane[i]
        return snap
    }

    /** Returns recent spike neuron IDs (thread-safe snapshot for Canvas). */
    fun getRecentSpikesSnapshot(): IntArray {
        for (i in 0 until spikeSnapshot.size) spikeSnapshot[i] = -1
        if (recentSpikeCount > 0) {
            val n = min(recentSpikeCount, spikeSnapshot.size)
            val start = if (recentSpikeWrite >= n) recentSpikeWrite - n else recentSpikeWrite + recentSpikes.size - n
            for (j in 0 until n) {
                spikeSnapshot[j] = recentSpikes[(start + j) % recentSpikes.size]
            }
        }
        return spikeSnapshot
    }

    fun getSpikeCount(): Int = spikeBuffer.size()
    fun getLayerInfo(): IntArray = intArrayOf(
        VISUAL_START, VISUAL_SIZE,
        AUDITORY_START, AUDITORY_SIZE,
        HIDDEN_START, HIDDEN_SIZE,
        MOTOR_START, MOTOR_SIZE
    )

    fun getActiveMotorCount(): Int {
        var c = 0
        for (i in 0 until MOTOR_SIZE) if (motorOutput[i] > 0) c++
        return c
    }

    /** Map motor neuron pattern to spoken word. */
    fun getMotorWord(): String {
        var active = 0
        for (i in 0 until MOTOR_SIZE) if (motorOutput[i] > 0) active++
        return when {
            active == 0 -> ""
            active in 1..4 -> "spark"
            active in 5..8 -> "neuron"
            active in 9..16 -> "learning"
            active in 17..24 -> "awake"
            active in 25..32 -> "transcending"
            else -> ""
        }
    }
}

// ===========================================================================
// Section 3: TextToSpeech Motor Controller
// ===========================================================================

class MotorController(private val tts: TextToSpeech?) {
    private var lastWord: String = ""

    fun update(engine: ChildBrainSNN) {
        val word = engine.getMotorWord()
        if (word.isNotEmpty() && word != lastWord) {
            lastWord = word
            tts?.stop()
            tts?.speak(
                word, TextToSpeech.QUEUE_FLUSH, null, "child_brain_${engine.simTime}"
            )
            Log.d("ChildBrainSNN", "TTS spoken: \"${word}\" at simTime=${engine.simTime}")
        }
        if (word.isEmpty()) lastWord = ""
    }
}

// ===========================================================================
// Section 4: CameraX Visual Diff Analyzer
// ===========================================================================

@OptIn(ExperimentalGetImage::class)
class VisualDiffAnalyzer {
    private val DOWNSAMPLE = 16     // 16x16 = 256 pixels -> 256 visual neurons
    private val prevLum: FloatArray = FloatArray(DOWNSAMPLE * DOWNSAMPLE)
    private val diffBuffer: FloatArray = FloatArray(DOWNSAMPLE * DOWNSAMPLE)
    private val yBytes: ByteArray = ByteArray(256 * 256)  // preallocated Y-plane buffer

    fun analyze(image: ImageProxy, engine: ChildBrainSNN) {
        val plane = image.planes[0]
        val buffer = plane.buffer
        val stride = plane.rowStride
        val width = image.width
        val height = image.height
        val stepX = maxOf(width / DOWNSAMPLE, 1)
        val stepY = maxOf(height / DOWNSAMPLE, 1)

        // Read Y plane into preallocated buffer (zero allocation)
        val remaining = minOf(buffer.remaining(), yBytes.size)
        if (remaining > 0) buffer.get(yBytes, 0, remaining)

        // Downsample luminance and compute diff from previous frame
        for (py in 0 until DOWNSAMPLE) {
            for (px in 0 until DOWNSAMPLE) {
                val sx = px * stepX
                val sy = py * stepY
                val offset = sy * stride + sx
                val lum = if (offset < remaining) (yBytes[offset].toInt() and 0xFF) / 255f else 0f
                val idx = py * DOWNSAMPLE + px
                diffBuffer[idx] = abs(lum - prevLum[idx])
                prevLum[idx] = lum
            }
        }
        image.close()
        engine.setVisualInput(diffBuffer, DOWNSAMPLE * DOWNSAMPLE)
    }

    fun reset() {
        for (i in prevLum.indices) prevLum[i] = 0f
    }
}

// ===========================================================================
// Section 5: Simulation Thread Controller
// ===========================================================================

class SNNController(
    private val engine: ChildBrainSNN,
    private val tts: TextToSpeech?
) : Thread("SNN-Simulation") {
    @Volatile private var running = false
    private val motorCtrl = MotorController(tts)
    private val dt = 1.0f  // 1ms timestep
    private val subSteps = 8  // 8ms simulation per frame (~60 FPS outer)
    private val targetIntervalMs = 16L

    override fun run() {
        var frameCount = 0
        var spikeAccumulator = 0
        while (running && !Thread.currentThread().isInterrupted) {
            val frameStart = System.currentTimeMillis()
            repeat(subSteps) {
                engine.simulateAudio()
                engine.step(dt)
                spikeAccumulator += engine.getSpikeCount()
            }
            motorCtrl.update(engine)
            val elapsed = System.currentTimeMillis() - frameStart
            val sleepMs = targetIntervalMs - elapsed
            if (sleepMs > 0) {
                try { sleep(sleepMs) } catch (_: InterruptedException) { break }
            }
            frameCount++
            if (frameCount >= 60) {
                frameCount = 0
                Log.d("ChildBrainSNN", "Spike count (last 1s): $spikeAccumulator")
                spikeAccumulator = 0
            }
        }
    }

    fun startSimulation() {
        if (!running) {
            running = true
            start()
        }
    }

    fun stopSimulation() {
        running = false
        interrupt()
    }
}

// ===========================================================================
// Section 6: CameraX Setup (Compose-compatible via AndroidView)
// ===========================================================================

@OptIn(ExperimentalGetImage::class)
@Composable
fun CameraPreviewWithAnalysis(
    lifecycleOwner: LifecycleOwner,
    engine: ChildBrainSNN,
    analyzer: VisualDiffAnalyzer,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current

    AndroidView(
        factory = { ctx ->
            val previewView = PreviewView(ctx).apply {
                layoutParams = android.widget.FrameLayout.LayoutParams(
                    android.widget.FrameLayout.LayoutParams.MATCH_PARENT,
                    android.widget.FrameLayout.LayoutParams.MATCH_PARENT
                )
                scaleType = PreviewView.ScaleType.FILL_CENTER
            }

            val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
            cameraProviderFuture.addListener({
                val cameraProvider = cameraProviderFuture.get()
                val preview = Preview.Builder()
                    .setTargetResolution(android.util.Size(256, 256))
                    .build()
                    .also { it.setSurfaceProvider(previewView.surfaceProvider) }

                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(android.util.Size(256, 256))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        it.setAnalyzer(
                            ContextCompat.getMainExecutor(ctx)
                        ) { imageProxy ->
                            try { analyzer.analyze(imageProxy, engine) } catch (e: Exception) {
                                imageProxy.close()
                            }
                        }
                    }

                val selector = CameraSelector.DEFAULT_BACK_CAMERA
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner, selector, preview, analysis
                )
            }, ContextCompat.getMainExecutor(ctx))

            previewView
        },
        modifier = modifier
    )
}

// ===========================================================================
// Section 7: Compose Real-time Visualizer Canvas
// ===========================================================================

@Composable
fun NeuralVisualizerCanvas(
    engine: ChildBrainSNN,
    modifier: Modifier = Modifier
) {
    var snapshot by remember { mutableStateOf(engine.snapshot()) }

    LaunchedEffect(Unit) {
        while (true) {
            delay(16L)
            snapshot = engine.snapshot()
        }
    }

    Canvas(modifier = modifier.fillMaxSize()) {
        val current = snapshot
        val cols = 16
        val cellW = size.width / cols
        val cellH = size.height / (current.size.toFloat() / cols + 1f)

        for (n in 0 until current.size) {
            val row = n / cols
            val col = n % cols

            val baseColor = when {
                n < engine.AUDITORY_START -> layerColors[0]
                n < engine.HIDDEN_START -> layerColors[1]
                n < engine.MOTOR_START -> layerColors[2]
                else -> layerColors[3]
            }

            val intensity = current[n].coerceIn(0f, 1f)
            val alpha = 0.15f + 0.85f * intensity

            drawRect(
                color = baseColor.copy(alpha = alpha),
                topLeft = Offset(col * cellW, row * cellH),
                size = Size(cellW, cellH)
            )
        }

        // Draw recent spike events as bright dots
        val spikes = engine.getRecentSpikesSnapshot()
        val count = min(spikes.size, engine.recentSpikeCount)
        for (i in 0 until count) {
            val nid = spikes[i]
            if (nid < 0 || nid >= current.size) continue
            val row = nid / cols
            val col = nid % cols
            val cx = col * cellW + cellW * 0.5f
            val cy = row * cellH + cellH * 0.5f
            drawCircle(
                color = Color.White,
                radius = 2.5f,
                center = Offset(cx, cy)
            )
        }
    }
}

// ===========================================================================
// Section 8: Stats Overlay
// ===========================================================================

@Composable
fun StatsOverlay(
    engine: ChildBrainSNN,
    modifier: Modifier = Modifier
) {
    var spikeCount by remember { mutableIntStateOf(0) }
    var motorActive by remember { mutableIntStateOf(0) }
    var simTimeSec by remember {
        mutableFloatStateOf(0f)
    }

    LaunchedEffect(Unit) {
        while (true) {
            delay(500)
            spikeCount = engine.getSpikeCount()
            motorActive = engine.getActiveMotorCount()
            simTimeSec = engine.simTime / 1000f
        }
    }

    Column(
        modifier = modifier
            .fillMaxWidth(0.5f)
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .height(96.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xAA000000))
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(10.dp),
                verticalArrangement = Arrangement.Center
            ) {
                Text(
                    "Child Brain SNN",
                    color = Color.White,
                    fontWeight = FontWeight.Bold,
                    fontSize = 15.sp
                )
                Text(
                    "Neurons: 480  |  Synapses: ${engine.connCount}",
                    color = Color(0xFFB3D9FF),
                    fontSize = 11.sp
                )
                Text(
                    "Sim time: ${String.format("%.1f", simTimeSec)}s",
                    color = Color(0xFFB3D9FF),
                    fontSize = 11.sp
                )
                Text(
                    "Spikes(frame): $spikeCount  |  Active motor: $motorActive/32",
                    color = Color(0xFFB3D9FF),
                    fontSize = 11.sp
                )
            }
        }

        // Layer legend
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            for (i in 0 until 4) {
                Card(
                    modifier = Modifier
                        .weight(1f)
                        .height(24.dp),
                    colors = CardDefaults.cardColors(containerColor = Color(0x33000000))
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(
                                brush = Brush.horizontalGradient(
                                    colors = listOf(layerColors[i].copy(0.6f), layerColors[i])
                                )
                            ),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            layerNames[i],
                            color = Color.White,
                            fontSize = 8.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }
        }
    }
}
