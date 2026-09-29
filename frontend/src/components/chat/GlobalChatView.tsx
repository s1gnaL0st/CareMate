import { useEffect, useRef } from 'react';
import { useGlobalStore } from '../../store/useGlobalStore';
import AgentStatusBubble from './AgentStatusBubble';
import ChatCardRenderer from './ChatCardRenderer';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const GlobalChatView = () => {
    const { messages } = useGlobalStore();
    const chatEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

    return (
        <div className="message-list">
            {messages.map(msg => {
                const hasTextOrGenerating = msg.text || msg.isGenerating;
                return <div key={msg.id}>
                    {msg.role === 'assistant' && (msg.isGenerating || (msg.steps && msg.steps.length > 0)) && <AgentStatusBubble steps={msg.steps || []} isGenerating={msg.isGenerating ?? false} />}
                    {msg.cards?.map((card, idx) => <div key={idx} className="mb-3"><ChatCardRenderer payload={card} /></div>)}
                    {hasTextOrGenerating && <div className={`message-row ${msg.role === 'user' ? 'user' : ''}`}>
                        <div className="message-avatar">{msg.role === 'user' ? '我' : 'AI'}</div>
                        <div className="message-content">
                            <p className="message-meta">{msg.role === 'user' ? '你' : 'CareMate'} · {msg.role === 'user' ? '刚刚' : '辅助回答'}</p>
                            <div className="message-bubble">
                                {msg.text ? <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown> : <span className="typing-indicator">● ● ●</span>}
                            </div>
                        </div>
                    </div>}
                </div>;
            })}
            <div ref={chatEndRef} />
        </div>
    );
};

export default GlobalChatView;
