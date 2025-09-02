import React, { useState } from "react";
import axios from "axios";
import "./App.css";
import user_icon from "./icons/user.jpeg"
import bot_icon from "./icons/bot.jpeg"
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";


function App() {
  const [message, setMessage] = useState("");
  const [chatHistory, setChatHistory] = useState<{ user: string; bot: string }[]>([]);
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = el.scrollHeight + "px";  // grow to content
    }
    setMessage(e.target.value);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    const newMessage = { user: message, bot: "" };
    setChatHistory((prev) => [...prev, newMessage]);

    setMessage("");

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    try {
      const response = await axios.post("http://localhost:5001/chat", {message});
      const botMessage = response.data.response;

      setChatHistory((prev) => {
        const updated = [...prev];
        updated[updated.length - 1].bot = botMessage;
        return [...updated];
      });
    } catch (error) {
      console.error("Error communicating with the backend", error);

      setChatHistory((prev) => {
        const updated = [...prev];
        updated[updated.length - 1].bot = "We're sorry, there was an issue with your request.";
        return updated;
      });
    } finally {
      setIsLoading(false);
    }

  };

  return (
    <div className="App">
      <h1>OVPRI AI Assistant</h1>
      <div className="chat-box">
  {chatHistory.map((chat, index) => (
    <div key={index}>
      <div className="chat-message user">
        <img src={user_icon} width="40" alt="user" />
        <p>{chat.user}</p>
      </div>
    {chat.bot ? (
      <div className="chat-message bot">
        <img src={bot_icon} width="40" alt="bot" />
        <div className="chat-bubble">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {chat.bot}
          </ReactMarkdown>
        </div>
      </div>
    ) : isLoading && index === chatHistory.length - 1 ? (
      // Only show thinking for the latest message being processed
      <div className="chat-message bot">
        <img src={bot_icon} width="40" alt="bot" />
        <div className="chat-bubble loading">
          Thinking<span className="dots"></span>
        </div>
      </div>
    ) : null}
    </div>
  ))}
</div>
      <form onSubmit={handleSubmit} className="chat-form">
        <textarea className="chat-input"
          ref={textareaRef}
          value={message}
          rows={1}
          onChange={handleChange}
          placeholder="Ask me about the IRB"
        />
        <button type="submit" className="chat-submit">Send</button>
      </form>
    </div>
  );
}

export default App;
