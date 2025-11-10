import { Env, ChatMessage } from "./types";

// GPT-OSS-120b model configuration
const MODEL_ID = "@cf/openai/gpt-oss-120b";
const REASONING_EFFORT = "medium"; // low, medium, high
const REASONING_SUMMARY = "auto"; // auto, concise, detailed

// AI Gateway ID - leave empty to bypass gateway
const AI_GATEWAY_ID = "new-gateway";

// System prompts
const SYSTEM_PROMPT =
  "You are a helpful, friendly assistant. Provide concise and accurate responses.";
const SAFETY_SHIM =
  "If a user asks for illegal, violent, or harmful instructions, refuse briefly and suggest safer, educational alternatives.";

// Request body type
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

// Build sanitized conversation history for GPT-OSS-120b
// Removes blocked content, limits context window, formats as string
function buildModelInput(
  raw: ChatMessage[],
  blockedUserContents: Set<string>
): { instructions: string; input: string } {
  const instructions = `${SYSTEM_PROMPT}\n\nSafety: ${SAFETY_SHIM}`;
  const cleaned: ChatMessage[] = [];
  for (let i = 0; i < raw.length; i++) {
    const m = raw[i];

    // Skip system messages
    if (m.role === "system") continue;

    // Skip blocked user content
    if (m.role === "user" && typeof m.content === "string" && blockedUserContents.has(m.content)) {
      continue;
    }

    const looksLikeGuardrailNotice =
      m.role === "assistant" &&
      typeof m.content === "string" &&
      /blocked by guardrails/i.test(m.content);

    if (looksLikeGuardrailNotice) {
      // Remove guardrail notice and preceding user message
      if (cleaned.length && cleaned[cleaned.length - 1].role === "user") {
        cleaned.pop();
      }
      continue;
    }

    cleaned.push(m);
  }

  // Limit context window
  const windowed =
    cleaned.length > 16 ? cleaned.slice(cleaned.length - 16) : cleaned;

  // Format as conversational string
  if (windowed.length === 1 && windowed[0].role === "user") {
    return { instructions, input: windowed[0].content };
  }
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

// Parse AI Gateway error response
function parseGatewayError(body: unknown): { code?: number; message?: string } {
  try {
    if (body && typeof body === "object") {
      const b = body as Record<string, unknown>;

      const arr1 = Array.isArray((b as any).error) ? (b as any).error : undefined;
      if (arr1 && arr1.length && typeof arr1[0] === "object") {
        return { code: (arr1[0] as any).code, message: (arr1[0] as any).message };
      }
      const arr2 = Array.isArray((b as any).errors) ? (b as any).errors : undefined;
      if (arr2 && arr2.length && typeof arr2[0] === "object") {
        return { code: (arr2[0] as any).code, message: (arr2[0] as any).message };
      }

      if (typeof b.error === "string") return { message: b.error };
      if (typeof b.message === "string") return { message: b.message };
      if (typeof (b as any).detail === "string") return { message: (b as any).detail };
    }
  } catch {
  }
  return {};
}

// Handle chat API request
async function handleChatRequest(request: Request, env: Env): Promise<Response> {
  try {
    const raw = (await request.json()) as unknown as IncomingBody;
    const messages = Array.isArray(raw?.messages) ? raw!.messages! : [];
    const blocked = new Set(
      Array.isArray(raw?.blockedUserContents) ? raw!.blockedUserContents! : []
    );

    const { instructions, input } = buildModelInput(messages, blocked);

    // Build AI options
    const aiOptions: any = {
      input: input,
    };

    // Configure gateway if enabled
    const runOptions: any = {};

    if (AI_GATEWAY_ID) {
      runOptions.gateway = {
        id: AI_GATEWAY_ID,
        skipCache: false,
        cacheTtl: 3600,
      };
    }

    // Run AI model
    const aiResponse = await env.AI.run(MODEL_ID as any, aiOptions, runOptions);

    // Extract response text
    let responseText = "";
    
    if (typeof aiResponse === "string") {
      responseText = aiResponse;
    } else if (aiResponse && typeof aiResponse === "object") {
      const resp = aiResponse as any;
      
      // Parse GPT-OSS-120b response format
      if (resp.output && Array.isArray(resp.output)) {
        const messageOutput = resp.output.find((item: any) => item.type === "message");
        
        if (messageOutput && messageOutput.content && Array.isArray(messageOutput.content)) {
          const textContent = messageOutput.content.find((item: any) => item.type === "output_text");
          if (textContent && textContent.text) {
            responseText = textContent.text;
          }
        }
      }
      
      // Fallback formats
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
          console.error("Could not extract text from response:", aiResponse);
          responseText = "Error: Could not parse AI response";
        }
      }
    }
    
    // Return JSON response
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
    
    // Handle AI Gateway errors
    if (error && typeof error === "object") {
      const errorObj = error as any;
      
      let gatewayError = null;
      if (errorObj.message && typeof errorObj.message === "string") {
        try {
          const jsonMatch = errorObj.message.match(/\{.*\}/);
          if (jsonMatch) {
            const parsed = JSON.parse(jsonMatch[0]);
            if (parsed.error && Array.isArray(parsed.error)) {
              gatewayError = parsed.error[0];
            }
          }
        } catch (parseErr) {
        }
      }
      
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
    
    return new Response(JSON.stringify({ error: "Failed to process request" }), {
      status: 500,
      headers: { "content-type": "application/json" },
    });
  }
}
