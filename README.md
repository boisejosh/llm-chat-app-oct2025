# Cloudflare AI Chat App

A chat application built with Cloudflare Workers AI and AI Gateway. Features streaming responses, markdown rendering, and optional content filtering through AI Gateway guardrails.

[![Deploy to Cloudflare Workers](https://deploy.workers.cloudflare.com/button)](https://deploy.workers.cloudflare.com/?url=https://github.com/acme-studios/chatbot-with-gateway)

## What This Does

This is a simple chatbot that uses Cloudflare Workers AI to generate responses. By default, it sends requests directly to the AI model. You can optionally route requests through AI Gateway to add content filtering, caching, and rate limiting.

The UI supports both light and dark themes, renders markdown responses with syntax highlighting, and handles errors gracefully.

## Quick Start

```bash
npm install
npm run dev
```

Visit http://localhost:8787 to test locally.

To deploy:
```bash
npm run deploy
```

## Changing the AI Model

Open `src/index.ts` and update the `MODEL_ID` constant:

```typescript
const MODEL_ID = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
```

Find available models at https://developers.cloudflare.com/workers-ai/models/

## Setting Up AI Gateway (Optional)

AI Gateway adds content filtering, caching, and analytics. Here's how to set it up:

### 1. Create an AI Gateway

1. Go to your Cloudflare dashboard
2. Navigate to AI > AI Gateway
3. Click "Create Gateway"
4. Give it a name (e.g., "chatbot-gateway")
5. Save the gateway

### 2. Configure Guardrails

In your gateway settings:

1. Go to the "Guardrails" tab
2. Enable the content filters you want:
   - Hate speech
   - Violence
   - Self-harm
   - Sexual content
   - etc.
3. Save your settings

### 3. Update Your Code

Open `src/index.ts` and set your gateway ID on line 9:

```typescript
const AI_GATEWAY_ID = "chatbot-gateway"; // Your gateway name
```

### 4. Redeploy

```bash
npm run deploy
```

Now all requests will go through your AI Gateway. Blocked prompts or responses will show detailed error messages in the UI.

## How It Works

- User sends a message
- If AI Gateway is configured, the request goes through the gateway first
- Gateway checks content against guardrails
- If approved, request goes to Workers AI model
- Response streams back to the user in real-time
- If blocked, user sees a detailed error message

Without AI Gateway configured, requests go directly to the model.

## Project Structure

```
src/index.ts       - Backend API and AI logic
src/types.ts       - TypeScript definitions
public/index.html  - UI and styling
public/chat.js     - Frontend logic
wrangler.jsonc     - Worker configuration
```

## Resources

- [Workers AI Models](https://developers.cloudflare.com/workers-ai/models/)
- [AI Gateway Documentation](https://developers.cloudflare.com/ai-gateway/)
- [Workers Documentation](https://developers.cloudflare.com/workers/)
