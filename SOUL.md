# SOUL.md — Child Brain SNN Engine

## Project Identity
- **Name:** ChildBrainSNN
- **Target:** Android 15 (API 35)
- **Domain:** Spiking Neural Network (SNN) engine for child brain modeling

## Core Constraints

### 1. Android 15 Target
- CompileSdk / targetSdk = 35
- Use AndroidX only; no legacy support libraries
- Minimize native dependencies; prefer pure Kotlin/JVM

### 2. Low-RAM Constraint (<2GB)
- JVM heap: `-Xmx1536m` (enforced in gradle.properties)
- No large batch allocations during runtime
- Stream data; avoid materializing full datasets in memory
- Recycle bitmaps and native buffers promptly

### 3. SNN Hebbian Learning Logic
- Spike-timing-dependent plasticity (STDP) is the primary learning rule
- Weight updates are local (pre/post spike pairs only — no global error signal)
- Synaptic decay applied per-frame to prevent saturation
- Neuron fire thresholds adapt via homeostatic scaling

### 4. Zero-Allocation in Loops
- Pre-allocate all buffers, lists, and objects before entering render/update loops
- Reuse object pools for spikes, events, and neuron state arrays
- Never call `new`/`alloc` inside the SNN tick loop or the Android render loop
- Profile allocations with Android Studio Memory Profiler; GC churn must stay <1MB/tick

### 5. Communication Protocol
- All responses are concise, direct, and factual
- No filler phrases, no restatement of the task
- Lead with the result; explain only when needed
- Code changes = summary of what changed + verification status

## Non-Negotiables
- Never commit secrets, .env files, or credential material
- Honor git boundaries: no history rewrites unless explicitly requested
- Every claim must be backed by real execution output (build, test, or measurement)
