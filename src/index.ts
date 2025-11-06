
import { Env, ChatMessage } from "./types";

// Model ID for Workers AI model
// GPT-OSS-120b is OpenAI's open-weight model designed for powerful reasoning and agentic tasks
const MODEL_ID = "@cf/openai/gpt-oss-120b";

// Reasoning configuration for GPT-OSS-120b
// effort: Controls computational effort on reasoning (low, medium, high)
// Higher effort results in more thorough reasoning but uses more tokens and time
const REASONING_EFFORT = "medium";

// summary: Controls the detail level of reasoning summaries (auto, concise, detailed)
// Useful for debugging and understanding the model's reasoning process
const REASONING_SUMMARY = "auto";

// AI Gateway Configuration (optional)
// Set your gateway ID to enable AI Gateway with guardrails, caching, and analytics
// Leave empty ("") to send requests directly to the model without AI Gateway
// When empty, all requests go straight to GPT-OSS-120b without any gateway processing
const AI_GATEWAY_ID = "new-gateway"; // Example: "chatbot-gateway" - Create an AI Gateway in the Dashboard and set the ID here

// Default system prompt (kept) + small safety shim
const SYSTEM_PROMPT =
  "You are a helpful, friendly assistant. Provide concise and accurate responses.";
const SAFETY_SHIM =
  "If a user asks for illegal, violent, or harmful instructions, refuse briefly and suggest safer, educational alternatives.";

/**
 * Normalize incoming JSON and be resilient to extra fields
 */
type IncomingBody = {
  messages?: ChatMessage[];
  blockedUserContents?: string[];
};

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    // Serve static frontend assets
    if (url.pathname === "/" || !url.pathname.startsWith("/api/")) {
      return env.ASSETS.fetch(request);
    }

    // Handle chat API route
    if (url.pathname === "/api/chat") {
      if (request.method !== "POST") {
        return new Response("Method not allowed", { status: 405 });
      }
      return handleChatRequest(request, env);
    }

    return new Response("Not found", { status: 404 });
  },
} satisfies ExportedHandler<Env>;

/**
 * Build sanitized conversation history for GPT-OSS-120b Responses API
 * 
 * GPT-OSS-120b uses the Responses API format which differs from the chat/messages API:
 * - Uses 'input' parameter (string or array) for the user's input
 * - Uses 'instructions' parameter for system prompts instead of system role messages
 * 
 * This function:
 * - Cleans and sanitizes the conversation history
 * - Drops any assistant lines that look like 'blocked by guardrails'
 * - Drops any user messages whose content is present in blockedUserContents
 * - Limits to last 16 messages to avoid context overflow
 * - Formats the conversation as a simple string for the input parameter
 */
function buildModelInput(
  raw: ChatMessage[],
  blockedUserContents: Set<string>
): { instructions: string; input: string } {
  // Build the instructions (system prompt) for the model
  const instructions = `${SYSTEM_PROMPT}\n\nSafety: ${SAFETY_SHIM}`;

  // Clean up historical turns
  const cleaned: ChatMessage[] = [];
  for (let i = 0; i < raw.length; i++) {
    const m = raw[i];

    // Ignore any system prompts coming from client (handled via instructions parameter)
    if (m.role === "system") continue;

    // Drop any user content we've previously marked as blocked
    if (m.role === "user" && typeof m.content === "string" && blockedUserContents.has(m.content)) {
      continue;
    }

    const looksLikeGuardrailNotice =
      m.role === "assistant" &&
      typeof m.content === "string" &&
      /blocked by guardrails/i.test(m.content);

    if (looksLikeGuardrailNotice) {
      // Drop this assistant message and (if adjacent) the previous user message
      if (cleaned.length && cleaned[cleaned.length - 1].role === "user") {
        cleaned.pop();
      }
      continue;
    }

    cleaned.push(m);
  }

  // Limit to last 16 messages to stay within context window efficiently
  const windowed =
    cleaned.length > 16 ? cleaned.slice(cleaned.length - 16) : cleaned;

  // Format conversation as a simple string
  // For GPT-OSS-120b, we'll format it as a conversational string
  // If there's only one message (the current user message), just return it
  if (windowed.length === 1 && windowed[0].role === "user") {
    return { instructions, input: windowed[0].content };
  }
  
  // If there's conversation history, format it as a readable conversation
  const conversationText = windowed
    .map((m) => {
      if (m.role === "user") {
        return `User: ${m.content}`;
      } else if (m.role === "assistant") {
        return `Assistant: ${m.content}`;
      }
      return "";
    })
    .filter(Boolean)
    .join("\n\n");

  return { instructions, input: conversationText };
}

/** Parse AI Gateway error shapes robustly (fixes TS squiggles) */
function parseGatewayError(body: unknown): { code?: number; message?: string } {
  try {
    if (body && typeof body === "object") {
      const b = body as Record<string, unknown>;

      // { error: [{ code, message }]} or { errors: [{ code, message }]}
      const arr1 = Array.isArray((b as any).error) ? (b as any).error : undefined;
      if (arr1 && arr1.length && typeof arr1[0] === "object") {
        return { code: (arr1[0] as any).code, message: (arr1[0] as any).message };
      }
      const arr2 = Array.isArray((b as any).errors) ? (b as any).errors : undefined;
      if (arr2 && arr2.length && typeof arr2[0] === "object") {
        return { code: (arr2[0] as any).code, message: (arr2[0] as any).message };
      }

      // { error: "string" } or { message: "string" } or { detail: "string" }
      if (typeof b.error === "string") return { message: b.error };
      if (typeof b.message === "string") return { message: b.message };
      if (typeof (b as any).detail === "string") return { message: (b as any).detail };
    }
  } catch {
    // fallthrough
  }
  return {};
}

/**
 * Handles the POST /api/chat request, calls the model, and streams the response
 */
async function handleChatRequest(request: Request, env: Env): Promise<Response> {
  try {
    const raw = (await request.json()) as unknown as IncomingBody;
    const messages = Array.isArray(raw?.messages) ? raw!.messages! : [];
    const blocked = new Set(
      Array.isArray(raw?.blockedUserContents) ? raw!.blockedUserContents! : []
    );

    // Build sanitized model input for GPT-OSS-120b Responses API
    const { instructions, input } = buildModelInput(messages, blocked);

    // Build AI options for GPT-OSS-120b Responses API
    // Note: GPT-OSS-120b uses 'input' (string) and 'instructions' instead of 'messages'
    const aiOptions: any = {
      // input: The conversation as a string (not an array)
      input: input,
    };
    
    // Add instructions if we have them
    // Note: Testing without instructions first to see if that's causing issues
    // Uncomment if needed:
    // aiOptions.instructions = instructions;
    
    // Add reasoning configuration if needed
    // Note: Commenting out temporarily to test basic functionality
    // Uncomment once basic chat is working:
    // aiOptions.reasoning = {
    //   effort: REASONING_EFFORT,
    //   summary: REASONING_SUMMARY,
    // };

    // Build run options with optional gateway
    // Note: Removing returnRawResponse to get the parsed response directly
    const runOptions: any = {};

    // Only add gateway if AI_GATEWAY_ID is configured
    if (AI_GATEWAY_ID) {
      runOptions.gateway = {
        id: AI_GATEWAY_ID,
        skipCache: false,
        cacheTtl: 3600,
      };
    }

    // Run LLM request (with or without AI Gateway)
    // Cast MODEL_ID to 'any' to avoid TypeScript errors with new model IDs not yet in type definitions
    const aiResponse = await env.AI.run(MODEL_ID as any, aiOptions, runOptions);

    // Extract the response text from the AI response
    // GPT-OSS-120b returns a complex response structure with reasoning and message output
    let responseText = "";
    
    if (typeof aiResponse === "string") {
      responseText = aiResponse;
    } else if (aiResponse && typeof aiResponse === "object") {
      const resp = aiResponse as any;
      
      // GPT-OSS-120b Responses API format:
      // The response has an "output" array with reasoning and message objects
      // We need to find the message object and extract the text from it
      if (resp.output && Array.isArray(resp.output)) {
        // Find the message output (type: "message")
        const messageOutput = resp.output.find((item: any) => item.type === "message");
        
        if (messageOutput && messageOutput.content && Array.isArray(messageOutput.content)) {
          // Extract text from the first content item
          const textContent = messageOutput.content.find((item: any) => item.type === "output_text");
          if (textContent && textContent.text) {
            responseText = textContent.text;
          }
        }
      }
      
      // Fallback to other common formats if the above didn't work
      if (!responseText) {
        if (resp.response) {
          responseText = resp.response;
        } else if (resp.content) {
          responseText = resp.content;
        } else if (resp.choices && resp.choices[0]?.message?.content) {
          responseText = resp.choices[0].message.content;
        } else if (resp.result && resp.result.response) {
          responseText = resp.result.response;
        } else {
          // If we can't find the text, log the whole response
          console.error("Could not extract text from response:", aiResponse);
          responseText = "Error: Could not parse AI response";
        }
      }
    }
    
    // Return simple JSON response (non-streaming for now)
    return new Response(
      JSON.stringify({ 
        response: responseText,
        success: true 
      }),
      {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }
    );
  } catch (error) {
    console.error("Error processing chat request:", error);
    
    // Check if this is an AI Gateway security block error
    if (error && typeof error === "object") {
      const errorObj = error as any;
      
      // Parse AI Gateway error from the error object
      let gatewayError = null;
      if (errorObj.message && typeof errorObj.message === "string") {
        try {
          // The error message might contain JSON
          const jsonMatch = errorObj.message.match(/\{.*\}/);
          if (jsonMatch) {
            const parsed = JSON.parse(jsonMatch[0]);
            if (parsed.error && Array.isArray(parsed.error)) {
              gatewayError = parsed.error[0];
            }
          }
        } catch (parseErr) {
          // Ignore parse errors
        }
      }
      
      // Handle specific AI Gateway error codes
      if (gatewayError && gatewayError.code === 2016) {
        return new Response(
          JSON.stringify({
            error: "Prompt Blocked by Security Policy",
            errorType: "prompt_blocked",
            details: AI_GATEWAY_ID
              ? "Your message was blocked by your organization's AI Gateway security policy. This may be due to content that violates safety guidelines including: hate speech, violence, self-harm, explicit content, or other harmful material."
              : "Your message was blocked due to security policy.",
            usingGateway: !!AI_GATEWAY_ID,
          }),
          {
            status: 400,
            headers: { "Content-Type": "application/json" },
          }
        );
      } else if (gatewayError && gatewayError.code === 2017) {
        return new Response(
          JSON.stringify({
            error: "Response Blocked by Security Policy",
            errorType: "response_blocked",
            details: AI_GATEWAY_ID
              ? "The AI's response was blocked by your organization's AI Gateway security policy. The model attempted to generate content that violates safety guidelines. Please rephrase your question or try a different topic."
              : "The AI's response was blocked due to security policy.",
            usingGateway: !!AI_GATEWAY_ID,
          }),
          {
            status: 400,
            headers: { "Content-Type": "application/json" },
          }
        );
      }
    }
    
    // Generic error fallback
    return new Response(JSON.stringify({ error: "Failed to process request" }), {
      status: 500,
      headers: { "content-type": "application/json" },
    });
  }
}
