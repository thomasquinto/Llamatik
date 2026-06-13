package com.llamatik.library.platform

import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext
import java.io.File

@Suppress("EXPECT_ACTUAL_CLASSIFIERS_ARE_IN_BETA_WARNING")
actual object LlamaBridge {
    private var nativeLibraryLoaded: Boolean = false

    init {
        nativeLibraryLoaded = try {
            System.loadLibrary("llama_jni") // Only here, where System exists
            true
        } catch (_: UnsatisfiedLinkError) {
            false
        }
    }

    actual external fun initModel(modelPath: String): Boolean
    actual external fun embed(input: String): FloatArray

    @Composable
    actual fun getModelPath(modelFileName: String): String {
        val context = LocalContext.current
        val outFile = File(context.cacheDir, modelFileName)

        if (!outFile.exists()) {
            context.assets.open(modelFileName).use { inputStream ->
                outFile.outputStream().use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
        }

        return outFile.absolutePath
    }

    actual external fun initGenerateModel(modelPath: String): Boolean
    actual external fun generate(prompt: String): String
    actual external fun generateWithContext(systemPrompt: String, contextBlock: String, userPrompt: String): String
    private external fun nativeGenerateStream(prompt: String, callback: GenStream)
    private external fun nativeGenerateWithContextStream(system: String, context: String, user: String, callback: GenStream)
    private external fun nativeUpdateGenerationParams(
        temperature: Float,
        maxTokens: Int,
        topP: Float,
        topK: Int,
        repeatPenalty: Float,
    )

    actual fun updateGenerateParams(
        temperature: Float,
        maxTokens: Int,
        topP: Float,
        topK: Int,
        repeatPenalty: Float,
    ) {
        nativeUpdateGenerationParams(temperature, maxTokens, topP, topK, repeatPenalty)
    }

    actual fun generateStream(prompt: String, callback: GenStream) {
        nativeGenerateStream(prompt, callback)
    }

    actual fun generateStreamWithContext(
        systemPrompt: String,
        contextBlock: String,
        userPrompt: String,
        callback: GenStream
    ) {
        // Delegate to native code which handles prompt formatting in a model-agnostic way
        nativeGenerateWithContextStream(systemPrompt, contextBlock, userPrompt, callback)
    }

    actual fun generateWithContextStream(
        system: String,
        context: String,
        user: String,
        onDelta: (String) -> Unit,
        onDone: () -> Unit,
        onError: (String) -> Unit
    ) {
        val cb = object : GenStream {
            override fun onDelta(text: String) = onDelta(text)
            override fun onComplete() = onDone()
            override fun onError(message: String) = onError(message)
        }
        nativeGenerateWithContextStream(system, context, user, cb)
    }

    // Native JNI function for message array streaming
    private external fun nativeGenerateWithMessagesStream(
        roles: Array<String>,
        contents: Array<String>,
        callback: GenStream
    )

    private external fun nativeIsVisionAvailable(): Boolean
    private external fun nativeInitVisionModel(modelPath: String, projectionModelPath: String): Boolean
    private external fun nativeGenerateVisionWithMessagesStream(
        roles: Array<String>,
        contents: Array<String>,
        imagePaths: Array<String>,
        callback: GenStream
    )

    actual fun generateStreamWithMessages(messages: List<ChatTemplateMessage>, callback: GenStream) {
        val roles = messages.map { it.role }.toTypedArray()
        val contents = messages.map { it.content }.toTypedArray()
        nativeGenerateWithMessagesStream(roles, contents, callback)
    }

    actual fun isVisionAvailable(): Boolean {
        return nativeLibraryLoaded && nativeIsVisionAvailable()
    }

    actual fun initVisionModel(modelPath: String, projectionModelPath: String): Boolean {
        return nativeLibraryLoaded && nativeInitVisionModel(modelPath, projectionModelPath)
    }

    actual fun generateVisionStreamWithMessages(
        messages: List<VisionChatTemplateMessage>,
        imagePaths: List<String>,
        callback: GenStream
    ) {
        if (!nativeLibraryLoaded) {
            callback.onError("llama.cpp native library is not available in this build.")
            return
        }
        val roles = messages.map { it.role }.toTypedArray()
        val contents = messages.map { it.content }.toTypedArray()
        nativeGenerateVisionWithMessagesStream(roles, contents, imagePaths.toTypedArray(), callback)
    }

    actual external fun shutdown()
    actual external fun nativeCancelGenerate()
}
