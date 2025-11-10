/**
 * UI behavior preserved; styling handled in styles.css
 * - Even spacing & dynamic bubble widths
 * - Smooth autoscroll on new messages
 * - Focus returns to textarea after send
 * - High-contrast pills now live in HTML as links
 */
const chatMessages = document.getElementById("chat-messages");
const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");
const typingIndicator = document.getElementById("typing-indicator");
const themeToggle = document.getElementById("theme-toggle");

let chatHistory = [
  { role: "assistant", content: "Hello! I'm an AI assistant powered by Cloudflare Workers AI. I can help you with questions, coding, writing, and more. How can I assist you today?" }
];

const blockedUserContents = [];
let isProcessing = false;
let lastSentUserText = "";

// Markdown highlighting config (unchanged)
if (typeof marked !== 'undefined') {
  marked.setOptions({
    breaks: true,
    gfm: true,
    highlight: function(code, lang) {
      if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
        try { return hljs.highlight(code, { language: lang }).value; } catch (err) {}
      }
      return code;
    }
  });
}

// Theme toggle (unchanged)
function initTheme() {
  const savedTheme = localStorage.getItem('theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);
  updateThemeIcon(savedTheme);
}
function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
  updateThemeIcon(next);
}
function updateThemeIcon(theme) {
  const sunIcon = themeToggle.querySelector('.sun-icon');
  const moonIcon = themeToggle.querySelector('.moon-icon');
  if (theme === 'dark') { sunIcon.style.display = 'block'; moonIcon.style.display = 'none'; }
  else { sunIcon.style.display = 'none'; moonIcon.style.display = 'block'; }
}
initTheme();
themeToggle.addEventListener('click', toggleTheme);

// Seed initial assistant bubble
renderMessage("assistant", chatHistory[0].content);

// Auto-resize textarea
userInput.addEventListener("input", function () {
  this.style.height = "auto";
  this.style.height = Math.min(this.scrollHeight, 200) + "px";
});

// Enter to send (Shift+Enter for newline)
userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendButton.addEventListener("click", sendMessage);

async function sendMessage() {
  const message = userInput.value.trim();
  if (!message || isProcessing) return;

  isProcessing = true;
  setInputsEnabled(false);
  typingIndicator.classList.add("visible");

  // user bubble
  renderMessage("user", message);
  userInput.value = "";
  userInput.style.height = "auto";

  lastSentUserText = message;
  chatHistory.push({ role: "user", content: message });

  // Assistant bubble we render into
  const assistantEl = renderMessage("assistant", "");
  const contentDiv = assistantEl.querySelector(".message-content");

  try {
    const sanitizedMessages = chatHistory.filter(
      (m) => !(m.role === "user" && blockedUserContents.includes(m.content))
    );

    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: sanitizedMessages, blockedUserContents })
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      assistantEl.remove();

      if (errorData.errorType === "prompt_blocked") {
        popLastUserTurn();
        rememberBlockedUser(lastSentUserText);
        renderErrorMessage(
          errorData.error || "Prompt Blocked",
          errorData.details || "Your message was blocked by security policy.",
          "prompt_blocked"
        );
      } else if (errorData.errorType === "response_blocked") {
        renderErrorMessage(
          errorData.error || "Response Blocked",
          errorData.details || "The AI's response was blocked by security policy.",
          "response_blocked"
        );
      } else {
        renderErrorMessage(
          errorData.error || "Error",
          errorData.details || "An error occurred while processing your request.",
          "general"
        );
      }
      return;
    }

    const data = await res.json();
    const responseText = data.response || "";

    if (typeof marked !== 'undefined') contentDiv.innerHTML = marked.parse(responseText);
    else contentDiv.textContent = responseText;

    scrollToBottom();
    chatHistory.push({ role: "assistant", content: responseText || "…" });
  } catch (err) {
    console.error(err);
    assistantEl.remove();
    renderErrorMessage(
      "Network Error",
      "Unable to connect to the server. Please check your connection and try again.",
      "network"
    );
  } finally {
    typingIndicator.classList.remove("visible");
    isProcessing = false;
    setInputsEnabled(true);
    userInput.focus(); // keep typing flow
  }
}

function renderMessage(role, content) {
  const wrap = document.createElement("div");
  wrap.className = `message ${role}-message`;
  const contentDiv = document.createElement("div");
  contentDiv.className = "message-content";

  if (role === "assistant" && typeof marked !== 'undefined' && content) {
    contentDiv.innerHTML = marked.parse(content);
  } else {
    contentDiv.textContent = content;
  }

  wrap.appendChild(contentDiv);
  chatMessages.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function renderErrorMessage(title, details) {
  const wrap = document.createElement("div");
  wrap.className = "error-message";

  const titleDiv = document.createElement("div");
  titleDiv.className = "error-title";

  const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  icon.setAttribute("class", "error-icon");
  icon.setAttribute("viewBox", "0 0 24 24");
  icon.setAttribute("fill", "currentColor");
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("d", "M12 22C6.47715 22 2 17.5228 2 12C2 6.47715 6.47715 2 12 2C17.5228 2 22 6.47715 22 12C22 17.5228 17.5228 22 12 22ZM12 20C16.4183 20 20 16.4183 20 12C20 7.58172 16.4183 4 12 4C7.58172 4 4 7.58172 4 12C4 16.4183 7.58172 20 12 20ZM11 15H13V17H11V15ZM11 7H13V13H11V7Z");
  icon.appendChild(path);

  titleDiv.appendChild(icon);
  titleDiv.appendChild(document.createTextNode(title));

  const detailsDiv = document.createElement("div");
  detailsDiv.className = "error-details";
  detailsDiv.textContent = details;

  wrap.appendChild(titleDiv);
  wrap.appendChild(detailsDiv);
  chatMessages.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function setInputsEnabled(enabled) {
  userInput.disabled = !enabled;
  sendButton.disabled = !enabled;
}

function popLastUserTurn() {
  for (let i = chatHistory.length - 1; i >= 0; i--) {
    if (chatHistory[i].role === "user") {
      chatHistory.splice(i, 1);
      return;
    }
  }
}

function rememberBlockedUser(text) {
  if (!text) return;
  if (!blockedUserContents.includes(text)) {
    blockedUserContents.push(text);
    if (blockedUserContents.length > 20) blockedUserContents.shift();
  }
}
