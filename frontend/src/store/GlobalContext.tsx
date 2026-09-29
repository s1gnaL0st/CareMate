// @refresh reset
import { createContext, useState, useCallback, useEffect, useRef, type ReactNode } from 'react';
import type { ChatMode, ChatMessage, ChatCardPayload, ScanType } from '../types';
import { createConversation, getAuthToken, listConversations, listMessages, type ConversationSummary } from '../services/chatService';

const LOCAL_CONVERSATIONS_KEY = 'smart_health_local_conversations';
const localMessagesKey = (id: string) => `smart_health_local_messages:${id}`;
const readLocalConversations = (): ConversationSummary[] => {
    try { return JSON.parse(localStorage.getItem(LOCAL_CONVERSATIONS_KEY) || '[]') as ConversationSummary[]; } catch { return []; }
};
const makeLocalConversation = (): ConversationSummary => ({
    id: `local-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    title: `新对话 ${new Date().toLocaleDateString('zh-CN')}`,
    active_agent: 'advisor_agent', status: 'active',
    created_at: new Date().toISOString(), updated_at: new Date().toISOString(),
});

// Per-mode config used to auto-generate Welcome and Exit cards
const MODE_CARD_CONFIG: Record<string, { title: string; description: string; exitTitle: string }> = {
    clinic: {
        title: '🩺 AI 辅助诊室已开启',
        description: '您好！我是您的专属线上医生助手。请告诉我您主要哪里不舒服？大概持续多久了？',
        exitTitle: '已结束本次问诊',
    },
    insurance: {
        title: '🏦 医保服务大厅已开启',
        description: '可以帮您查询医保余额、消费记录、缴费明细等，请问需要什么帮助？',
        exitTitle: '医保咨询已结束',
    },
    pharmacy: {
        title: '💊 药管家模式已开启',
        description: '您可以上传药盒图片，或直接告诉我药品名称，我来帮您查询药效、注意事项和附近药店。',
        exitTitle: '药管家服务已结束',
    },
    report: {
        title: '📋 报告解读模式已开启',
        description: '请发送您的检查报告图片，或直接描述检查指标数值，我来为您进行 AI 解读。',
        exitTitle: '报告解读已结束',
    },
};

export interface GlobalState {
    isElderMode: boolean;
    setIsElderMode: (val: boolean | ((prev: boolean) => boolean)) => void;

    chatMode: ChatMode;
    setChatMode: (mode: ChatMode) => void;
    /** Enter a mode and inject a Welcome Card into the chat */
    enterChatMode: (mode: ChatMode) => void;
    /** Exit current mode and inject an Exit Card into the chat */
    exitChatMode: () => void;

    messages: ChatMessage[];
    setMessages: (msgs: ChatMessage[] | ((prev: ChatMessage[]) => ChatMessage[])) => void;

    isScanning: boolean;
    setIsScanning: (val: boolean) => void;
    scanType: ScanType;
    setScanType: (type: ScanType) => void;
    accessToken: string | null;
    setAccessToken: (token: string | null) => void;
    conversationId: string | null;
    setConversationId: (id: string | null) => void;
    conversations: ConversationSummary[];
    selectConversation: (id: string) => Promise<void>;
    createNewConversation: () => Promise<void>;
    conversationError: string | null;
}

const GlobalContext = createContext<GlobalState | undefined>(undefined);

export const GlobalProvider = ({ children }: { children: ReactNode }) => {
    const [isElderMode, setIsElderMode] = useState(false);
    const [chatMode, setChatMode] = useState<ChatMode>('general');
    const chatModeRef = useRef<ChatMode>('general');

    const createWelcomeMessage = () => ({
        id: `msg-welcome-${Date.now()}`,
        role: 'assistant' as const,
        text: '你好！我是 CareMate，你的智能健康伙伴。今天有什么我可以帮你的吗？',
        timestamp: Date.now(),
    });
    const [messages, setMessages] = useState<ChatMessage[]>(() => [createWelcomeMessage()]);

    const [isScanning, setIsScanning] = useState(false);
    const [accessToken, setAccessToken] = useState<string | null>(() => getAuthToken());
    const [conversationId, setConversationId] = useState<string | null>(() => localStorage.getItem('smart_health_conversation_id'));
    const [conversations, setConversations] = useState<ConversationSummary[]>([]);
    const [conversationError, setConversationError] = useState<string | null>(null);

    const selectConversation = useCallback(async (id: string) => {
        setConversationError(null);
        let persisted: Array<{ id: string; role: 'user' | 'assistant'; content: string; created_at: string }> = [];
        if (id.startsWith('local-')) {
            try { persisted = JSON.parse(localStorage.getItem(localMessagesKey(id)) || '[]'); } catch { persisted = []; }
        } else {
            persisted = await listMessages(id);
        }
        setConversationId(id);
        localStorage.setItem('smart_health_conversation_id', id);
        setChatMode('general');
        chatModeRef.current = 'general';
        setMessages(persisted.length > 0 ? persisted.map(message => ({
            id: message.id,
            role: message.role,
            text: message.content,
            timestamp: Date.parse(message.created_at),
        })) : [createWelcomeMessage()]);
    }, []);

    const createNewConversation = useCallback(async () => {
        setConversationError(null);
        let conversation: ConversationSummary;
        if (accessToken) {
            try {
                conversation = await createConversation(`新对话 ${new Date().toLocaleDateString('zh-CN')}`);
            } catch {
                setConversationError('云端会话创建失败，已创建本地会话');
                conversation = makeLocalConversation();
            }
        } else {
            conversation = makeLocalConversation();
        }
        if (conversation.id.startsWith('local-')) {
            const local = [conversation, ...readLocalConversations().filter(item => item.id !== conversation.id)];
            localStorage.setItem(LOCAL_CONVERSATIONS_KEY, JSON.stringify(local));
        }
        setConversations(current => [conversation, ...current.filter(item => item.id !== conversation.id)]);
        setConversationId(conversation.id);
        localStorage.setItem('smart_health_conversation_id', conversation.id);
        setChatMode('general');
        chatModeRef.current = 'general';
        setMessages([createWelcomeMessage()]);
    }, [accessToken]);

    useEffect(() => {
        if (!accessToken) {
            const local = readLocalConversations();
            const current = local.find(item => item.id === localStorage.getItem('smart_health_conversation_id')) ?? local[0] ?? makeLocalConversation();
            if (!local.length) localStorage.setItem(LOCAL_CONVERSATIONS_KEY, JSON.stringify([current]));
            setConversations(local.length ? local : [current]);
            void selectConversation(current.id);
            return;
        }
        let cancelled = false;
        const restore = async () => {
            try {
                const conversations = await listConversations();
                setConversations(conversations);
                const storedId = localStorage.getItem('smart_health_conversation_id');
                let conversation = conversations.find(item => item.id === storedId) ?? conversations[0];
                if (!conversation) {
                    conversation = await createConversation();
                    setConversations([conversation]);
                }
                if (cancelled) return;
                await selectConversation(conversation.id);
            } catch (error) {
                setConversationError('云端会话暂不可用，已切换为本地会话');
                console.warn('[store] failed to restore conversation', error);
            }
        };
        void restore();
        return () => { cancelled = true; };
    }, [accessToken, selectConversation]);
    useEffect(() => {
        if (!conversationId?.startsWith('local-')) return;
        const persisted = messages.filter(message => message.text).map(message => ({
            id: message.id, role: message.role === 'user' ? 'user' : 'assistant', content: message.text,
            created_at: new Date(message.timestamp).toISOString(),
        }));
        localStorage.setItem(localMessagesKey(conversationId), JSON.stringify(persisted));
    }, [conversationId, messages]);
    const [scanType, setScanType] = useState<ScanType>('药盒');

    const enterChatMode = useCallback((mode: ChatMode) => {
        if (mode === 'general' || mode === 'dashboard') {
            setChatMode(mode);
            chatModeRef.current = mode;
            return;
        }

        // Use ref to prevent duplicate triggers (Strict Mode safe)
        if (chatModeRef.current === mode) return;

        chatModeRef.current = mode;
        setChatMode(mode);

        const config = MODE_CARD_CONFIG[mode];
        if (config) {
            const welcomeCard: ChatCardPayload = {
                type: 'mode_welcome',
                mode,
                title: config.title,
                description: config.description,
            };

            setMessages(msgs => {
                const newMsgs = [...msgs];
                if (newMsgs.length > 0) {
                    const lastMsg = newMsgs[newMsgs.length - 1];
                    // If the last message is actively being generated by the assistant,
                    // attach the welcome card directly to it so it renders above the text bubble.
                    if (lastMsg.role === 'assistant' && lastMsg.isGenerating) {
                        return [
                            ...newMsgs.slice(0, -1),
                            {
                                ...lastMsg,
                                cards: [...(lastMsg.cards || []), welcomeCard]
                            }
                        ];
                    }
                }

                // Fallback: append as a separate message
                return [
                    ...newMsgs,
                    {
                        id: `system-welcome-${mode}-${Date.now()}`,
                        role: 'assistant',
                        text: '',
                        timestamp: Date.now(),
                        cards: [welcomeCard],
                    }
                ];
            });
        }
    }, []);

    const exitChatMode = useCallback(() => {
        const mode = chatModeRef.current;
        if (mode === 'general' || mode === 'dashboard') return;

        chatModeRef.current = 'general';
        setChatMode('general');

        const config = MODE_CARD_CONFIG[mode];
        if (config) {
            const exitCard: ChatCardPayload = {
                type: 'mode_exit',
                mode,
                title: config.exitTitle,
            };

            setMessages(msgs => {
                const newMsgs = [...msgs];
                if (newMsgs.length > 0) {
                    const lastMsg = newMsgs[newMsgs.length - 1];
                    // If the last message is actively being generated by the assistant,
                    // attach the exit card directly to it.
                    if (lastMsg.role === 'assistant' && lastMsg.isGenerating) {
                        return [
                            ...newMsgs.slice(0, -1),
                            {
                                ...lastMsg,
                                cards: [...(lastMsg.cards || []), exitCard]
                            }
                        ];
                    }
                }

                // Fallback: append as a separate message
                return [
                    ...newMsgs,
                    {
                        id: `system-exit-${mode}-${Date.now()}`,
                        role: 'assistant',
                        text: '',
                        timestamp: Date.now(),
                        cards: [exitCard]
                    }
                ];
            });
        }
    }, []);

    return (
        <GlobalContext.Provider value={{
            isElderMode, setIsElderMode,
            chatMode, setChatMode, enterChatMode, exitChatMode,
            messages, setMessages,
            isScanning, setIsScanning,
            scanType, setScanType,
            accessToken, setAccessToken, conversationId, setConversationId,
            conversations, selectConversation, createNewConversation, conversationError
        }}>
            {children}
        </GlobalContext.Provider>
    );
};

export { GlobalContext };
