package com.llamatik.library.platform

import androidx.compose.runtime.Composable

@Suppress("EXPECT_ACTUAL_CLASSIFIERS_ARE_IN_BETA_WARNING")
expect object LlamaBridge {
    @Composable
    fun getModelPath(modelFileName: String): String

    fun initModel(modelPath: String): Boolean
    fun embed(input: String): FloatArray
    fun initGenerateModel(modelPath: String): Boolean
    fun generate(prompt: String): String
    fun generateWithContext(systemPrompt: String, contextBlock: String, userPrompt: String): String
    fun generateStream(prompt: String, callback: GenStream)
    fun generateStreamWithContext(systemPrompt: String, contextBlock: String, userPrompt: String, callback: GenStream)
    fun generateWithContextStream(
        system: String,
        context: String,
        user: String,
        onDelta: (String) -> Unit,
        onDone: () -> Unit,
        onError: (String) -> Unit
    )

    /**
     * Generate streaming response using chat template with message array.
     * Uses the model's embedded chat template if available, otherwise falls back to ChatML.
     *
     * @param messages List of ChatTemplateMessage (role + content pairs)
     * @param callback GenStream callback for streaming tokens
     */
    fun generateStreamWithMessages(messages: List<ChatTemplateMessage>, callback: GenStream)

    fun shutdown()
    fun nativeCancelGenerate()
    fun updateGenerateParams(
        temperature: Float,
        maxTokens: Int,
        topP: Float,
        topK: Int,
        repeatPenalty: Float,
    )
}

/**
 * A message for chat template formatting.
 * @param role The role: "system", "user", or "assistant"
 * @param content The message content
 */
data class ChatTemplateMessage(
    val role: String,
    val content: String
) {
    companion object {
        fun system(content: String) = ChatTemplateMessage("system", content)
        fun user(content: String) = ChatTemplateMessage("user", content)
        fun assistant(content: String) = ChatTemplateMessage("assistant", content)
    }
}

interface GenStream {
    fun onDelta(text: String)
    fun onComplete()
    fun onError(message: String)
}
