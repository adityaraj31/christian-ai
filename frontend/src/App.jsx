import { useState, useRef, useEffect } from 'react'

const DENOMINATIONS = ["Protestant", "Catholic", "Orthodox"]

function sessionId() {
  let id = sessionStorage.getItem("chat_session_id")
  if (!id) {
    id = crypto.randomUUID()
    sessionStorage.setItem("chat_session_id", id)
  }
  return id
}

function ChatMessage({ role, text, citations, onVisualize, generating }) {
  const [showCitations, setShowCitations] = useState(false)
  const isUser = role === "user"

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div
        className={`max-w-[75%] rounded-2xl px-4 py-3 ${
          isUser
            ? "bg-blue-600 text-white rounded-br-md"
            : "bg-gray-100 text-gray-900 rounded-bl-md"
        }`}
      >
        {generating ? (
          <div className="flex items-center gap-2 py-2">
            <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
            <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
            <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
          </div>
        ) : (
          <>
            <p className="text-sm leading-relaxed whitespace-pre-wrap">{text}</p>
            {!isUser && citations && citations.length > 0 && (
              <div className="mt-2 pt-2 border-t border-gray-200/60">
                <button
                  onClick={() => setShowCitations(!showCitations)}
                  className="text-xs text-blue-500 hover:text-blue-700 font-medium"
                >
                  {showCitations ? "Hide" : "Show"} citations ({citations.length})
                </button>
                {showCitations && (
                  <div className="mt-2 space-y-1">
                    {citations.map((c, i) => (
                      <div key={i} className="text-xs bg-gray-50 rounded p-2 border border-gray-200">
                        <span className="font-semibold text-gray-600">{c.reference}</span>
                        <p className="text-gray-500 mt-0.5">{c.text}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
            {!isUser && text && text.length > 10 && (
              <button
                onClick={() => onVisualize(text)}
                className="mt-2 text-xs text-purple-500 hover:text-purple-700 font-medium"
              >
                Visualize this Context
              </button>
            )}
          </>
        )}
      </div>
    </div>
  )
}

function Sidebar({ denomination, onDenominationChange }) {
  return (
    <aside className="w-64 bg-white border-r border-gray-200 flex flex-col h-full">
      <div className="p-4 border-b border-gray-200">
        <h1 className="text-lg font-bold text-gray-800">Christian AI</h1>
        <p className="text-xs text-gray-500 mt-0.5">Grounded Biblical Assistant</p>
      </div>
      <div className="p-4">
        <label className="text-xs font-semibold text-gray-600 uppercase tracking-wider">
          Denomination
        </label>
        <select
          value={denomination}
          onChange={(e) => onDenominationChange(e.target.value)}
          className="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
        >
          {DENOMINATIONS.map((d) => (
            <option key={d} value={d}>{d}</option>
          ))}
        </select>
        <p className="text-xs text-gray-400 mt-2">
          Responses will reflect the selected tradition.
        </p>
      </div>
      <div className="mt-auto p-4 border-t border-gray-200">
        <div className="flex items-center gap-2 text-xs text-gray-400">
          <span className="w-2 h-2 bg-green-400 rounded-full" />
          Secure &amp; grounded
        </div>
      </div>
    </aside>
  )
}

export default function App() {
  const [messages, setMessages] = useState([
    { role: "assistant", text: "Hello! I'm your Christian Bible study assistant. Ask me anything about Scripture.", citations: [] },
  ])
  const [input, setInput] = useState("")
  const [denomination, setDenomination] = useState("Protestant")
  const [generating, setGenerating] = useState(false)
  const chatEnd = useRef(null)
  const sid = useRef(sessionId())

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  async function handleSend(e) {
    e.preventDefault()
    const trimmed = input.trim()
    if (!trimmed || generating) return

    const userMsg = { role: "user", text: trimmed }
    setMessages((prev) => [...prev, userMsg])
    setInput("")
    setGenerating(true)

    setMessages((prev) => [...prev, { role: "assistant", text: "", citations: [], generating: true }])

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed, denomination, session_id: sid.current }),
      })
      const data = await res.json()
      setMessages((prev) => {
        const copy = [...prev]
        copy[copy.length - 1] = {
          role: "assistant",
          text: data.response,
          citations: data.citations || [],
          safetyTriggered: data.safety_triggered,
        }
        return copy
      })
    } catch {
      setMessages((prev) => {
        const copy = [...prev]
        copy[copy.length - 1] = {
          role: "assistant",
          text: "Sorry, I encountered an error. Please try again.",
          citations: [],
        }
        return copy
      })
    } finally {
      setGenerating(false)
    }
  }

  async function handleVisualize(text) {
    try {
      const res = await fetch("/api/generate-image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.slice(0, 500) }),
      })
      const data = await res.json()
      if (data.image_url) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            text: `![Generated biblical scene](${data.image_url})`,
            citations: [],
            imageUrl: data.image_url,
          },
        ])
      } else {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", text: "Image generation was blocked by safety moderation.", citations: [] },
        ])
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "Image generation failed. Check API configuration.", citations: [] },
      ])
    }
  }

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar denomination={denomination} onDenominationChange={setDenomination} />
      <main className="flex-1 flex flex-col">
        <div className="flex-1 overflow-y-auto px-4 py-6">
          <div className="max-w-3xl mx-auto">
            {messages.map((msg, i) =>
              msg.imageUrl ? (
                <div key={i} className="flex justify-start mb-4">
                  <div className="max-w-[75%] rounded-2xl px-4 py-3 bg-gray-100 rounded-bl-md">
                    <img src={msg.imageUrl} alt="Generated biblical scene" className="rounded-lg w-full" />
                  </div>
                </div>
              ) : (
                <ChatMessage
                  key={i}
                  role={msg.role}
                  text={msg.text}
                  citations={msg.citations}
                  generating={msg.generating}
                  onVisualize={handleVisualize}
                />
              )
            )}
            <div ref={chatEnd} />
          </div>
        </div>
        <form onSubmit={handleSend} className="border-t border-gray-200 bg-white p-4">
          <div className="max-w-3xl mx-auto flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about the Bible..."
              disabled={generating}
              className="flex-1 rounded-xl border border-gray-300 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={generating || !input.trim()}
              className="bg-blue-600 text-white rounded-xl px-6 py-3 text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Send
            </button>
          </div>
        </form>
      </main>
    </div>
  )
}
