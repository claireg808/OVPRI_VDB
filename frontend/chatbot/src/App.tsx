import { useState } from "react";
import axios from "axios";
import "./App.css";
import user_icon from "./icons/user.jpeg"
import bot_icon from "./icons/bot.jpeg"

function App() {
  const [message, setMessage] = useState("");
  const [chatHistory, setChatHistory] = useState<{ user: string; bot: string }[]>([]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const newMessage = { user: message, bot: "" };
    setChatHistory((prev) => [...prev, newMessage]);

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
    }

    setMessage("");
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
      {chat.bot && (
        <div className="chat-message bot">
          <img src={bot_icon} width="40" alt="bot" />
          <p>{chat.bot}</p>
        </div>
      )}
    </div>
  ))}
</div>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Type your message..."
        />
        <button type="submit">Send</button>
      </form>
    </div>
  );
}

export default App;
