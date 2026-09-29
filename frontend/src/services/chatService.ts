import type { ChatMessage, AgentStep, ChatMode, ChatCardPayload } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

export const getAuthToken = () => localStorage.getItem('smart_health_access_token');
export const setAuthToken = (token: string | null) => {
    if (token) localStorage.setItem('smart_health_access_token', token);
    else localStorage.removeItem('smart_health_access_token');
};

export interface AuthUser {
    id: string;
    email: string;
    name: string;
    is_active: boolean;
}

interface AuthResponse {
    access_token: string;
    token_type: string;
    expires_in: number;
    user: AuthUser;
}

const apiJson = async <T>(path: string, init: RequestInit = {}, token?: string): Promise<T> => {
    const headers = new Headers(init.headers);
    headers.set('Content-Type', 'application/json');
    const authToken = token ?? getAuthToken();
    if (authToken) headers.set('Authorization', `Bearer ${authToken}`);
    const response = await fetch(`${API_BASE_URL}${path}`, { ...init, headers });
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return response.json() as Promise<T>;
};

export const login = async (email: string, password: string) => {
    const result = await apiJson<AuthResponse>('/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) });
    setAuthToken(result.access_token);
    return result;
};

export const register = async (email: string, password: string, name: string) => {
    const result = await apiJson<AuthResponse>('/auth/register', { method: 'POST', body: JSON.stringify({ email, password, name }) });
    setAuthToken(result.access_token);
    return result;
};
export const logout = async () => {
    try { await apiJson('/auth/logout', { method: 'POST' }); } finally { setAuthToken(null); }
};

export interface ConversationSummary {
    id: string;
    title: string;
    active_agent: string;
    status: string;
    created_at: string;
    updated_at: string;
}

export interface PersistedMessage {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    sequence: number;
    created_at: string;
}

export const listConversations = () => apiJson<ConversationSummary[]>('/conversations');
export const createConversation = (title = '新对话') => apiJson<ConversationSummary>('/conversations', { method: 'POST', body: JSON.stringify({ title }) });
export const listMessages = (conversationId: string) => apiJson<PersistedMessage[]>(`/conversations/${conversationId}/messages`);

export interface ReportListItem {
    report_id: string;
    original_filename: string;
    content_type: string;
    size_bytes: number;
    status: string;
    analysis_status?: string | null;
    created_at: string;
}

export const listReports = () => apiJson<ReportListItem[]>('/reports');

export interface HospitalItem {
    id: string; name: string; city: string; address: string; tags: string[];
    distance_km: number | null; phone: string; is_ad: boolean;
}
export const listHospitals = (params: { city?: string; q?: string } = {}) => {
    const query = new URLSearchParams();
    if (params.city) query.set('city', params.city);
    if (params.q) query.set('q', params.q);
    return apiJson<HospitalItem[]>(`/hospitals${query.toString() ? `?${query}` : ''}`);
};

export interface HabitGoal {
    id: string; title: string; target_value: number; current_value: number;
    unit: string; color: string; status: string; created_at: string; updated_at: string;
}
export const listHabitGoals = () => apiJson<HabitGoal[]>('/habit-goals');
export const updateHabitGoal = (id: string, current_value: number) => apiJson<HabitGoal>(`/habit-goals/${id}`, { method: 'PATCH', body: JSON.stringify({ current_value }) });

export const submitFeedback = (payload: { message_id?: string; conversation_id?: string; rating: -1 | 0 | 1; category?: string; comment?: string }) => apiJson('/feedback', { method: 'POST', body: JSON.stringify(payload) });

/** Maps LangGraph node names to ChatMode strings */
const NODE_TO_CHAT_MODE: Record<string, ChatMode> = {
    clinic_node: 'clinic',
    insurance_node: 'insurance',
    report_node: 'report',
    pharmacy_node: 'pharmacy',
    // advisor_node returning 'general' signals the user is exiting a specialized mode
    advisor_node: 'general',
};

export interface ChatServiceOptions {
    onChunk: (text: string) => void;
    onStep: (step: AgentStep) => void;
    onStepFinish: (nodeOrTool: string) => void;
    /** Called when LangGraph enters a specialized agent node, or advisor_node (general) to exit a mode */
    onModeChange?: (mode: ChatMode) => void;
    /** Called when a backend tool yields a structured UI card payload */
    onCard?: (card: ChatCardPayload) => void;
    onDone: () => void;
    onError: (error: Error) => void;
    conversationId?: string;
    token?: string;
}

export interface UserInfoPayload {
    name?: string;
    age?: number;
    medical_history?: string;
    elder_mode?: boolean;
    region?: string;
    profession?: string;
}

const LOCAL_PROFILE_KEY = 'smart_health_user_profile';
export const getLocalUserProfile = (): UserInfoPayload => {
    try { return JSON.parse(localStorage.getItem(LOCAL_PROFILE_KEY) || '{}') as UserInfoPayload; } catch { return {}; }
};
export const updateLocalUserProfileFromText = (text: string): UserInfoPayload => {
    const current = getLocalUserProfile();
    if (/我(?:是|是一名|的职业是)程序员|做程序员/.test(text)) current.profession = '程序员';
    if (/我(?:是|是一名|的职业是)医生/.test(text)) current.profession = '医生';
    if (/我(?:是|是一名|的职业是)护士/.test(text)) current.profession = '护士';
    if (/我(?:是|是一名|的职业是)药师/.test(text)) current.profession = '药师';
    localStorage.setItem(LOCAL_PROFILE_KEY, JSON.stringify(current));
    return current;
};

export type VisionScanType = 'report' | 'drug_box' | 'trace_code';

interface ReportUploadResponse {
    report_id: string;
    analysis_id: string;
    status: string;
    duplicate: boolean;
}

interface ReportStatusResponse {
    report_id: string;
    status: string;
    analysis_status: string | null;
    analysis: { text?: string } | null;
    error: string | null;
    attempt_count: number;
}

let _stepCounter = 0;
const genStepId = () => `step-${++_stepCounter}-${Date.now()}`;

const consumeSseResponse = async (response: Response, options: ChatServiceOptions) => {
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    if (!response.body) {
        throw new Error("ReadableStream not yet supported in this browser.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let done = false;
    let buffer = '';

    while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;

        if (value) {
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            // Keep the last (potentially incomplete) line in the buffer
            buffer = lines.pop() ?? '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const jsonStr = line.slice(6).trim();
                if (!jsonStr) continue;

                try {
                    const data = JSON.parse(jsonStr);

                    switch (data.type) {
                        case 'text':
                            options.onChunk(data.content ?? '');
                            break;

                        case 'node_start': {
                            const step: AgentStep = {
                                id: genStepId(),
                                type: 'node_start',
                                node: data.node,
                                content: data.content ?? `进入节点：${data.node}`,
                                isFinished: false,
                            };
                            options.onStep(step);
                            // Automatically switch chatMode based on which agent node started
                            console.log('[chatService] node_start:', data.node, '→ mode:', NODE_TO_CHAT_MODE[data.node] ?? '(no mapping)');
                            if (data.node && NODE_TO_CHAT_MODE[data.node] && options.onModeChange) {
                                console.log('[chatService] calling onModeChange:', NODE_TO_CHAT_MODE[data.node]);
                                options.onModeChange(NODE_TO_CHAT_MODE[data.node]);
                            }
                            break;
                        }

                        case 'node_end':
                            options.onStepFinish(data.node ?? '');
                            break;

                        case 'tool_start': {
                            const step: AgentStep = {
                                id: genStepId(),
                                type: 'tool_start',
                                tool: data.tool,
                                content: data.content ?? `调用工具：${data.tool}`,
                                isFinished: false,
                            };
                            options.onStep(step);
                            break;
                        }

                        case 'tool_end':
                            options.onStepFinish(data.tool ?? '');
                            break;

                        case 'card':
                            if (options.onCard && data.payload) {
                                options.onCard(data.payload as ChatCardPayload);
                            }
                            break;

                        case 'finish':
                            done = true;
                            break;

                        case 'error':
                            options.onError(new Error(data.content ?? '未知错误'));
                            done = true;
                            break;
                    }
                } catch {
                    // Ignore malformed JSON lines
                }
            }
        }
    }
};

export const streamChat = async (
    messages: ChatMessage[],
    options: ChatServiceOptions,
    userInfo?: UserInfoPayload,
    chatMode: ChatMode = 'general'
) => {
    try {
        const payload = {
            messages: messages.map(msg => ({
                role: msg.role,
                content: msg.text
            })),
            user_info: userInfo ?? {},
            chat_mode: chatMode,
            conversation_id: options.conversationId,
            idempotency_key: `web-${Date.now()}-${Math.random().toString(36).slice(2)}`,
        };

        const headers = new Headers({ 'Content-Type': 'application/json' });
        const token = options.token ?? getAuthToken();
        if (token) headers.set('Authorization', `Bearer ${token}`);
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers,
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        await consumeSseResponse(response, options);

        options.onDone();

    } catch (err) {
        options.onError(err instanceof Error ? err : new Error(String(err)));
    }
};

export const streamVisionChat = async (
    file: File,
    scanType: VisionScanType,
    messages: ChatMessage[],
    options: ChatServiceOptions,
    userInfo?: UserInfoPayload,
) => {
    try {
        const token = options.token ?? getAuthToken();
        if (scanType === 'report' && token) {
            await streamBackgroundReportAnalysis(file, scanType, token, options);
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        formData.append('scan_type', scanType);
        formData.append('user_info', JSON.stringify(userInfo ?? {}));
        formData.append('messages', JSON.stringify(messages.map(msg => ({
            role: msg.role,
            content: msg.text
        }))));

        const headers = new Headers();
        if (token) headers.set('Authorization', `Bearer ${token}`);
        const response = await fetch(`${API_BASE_URL}/vision-chat`, {
            method: 'POST',
            headers,
            body: formData
        });

        await consumeSseResponse(response, options);
        options.onDone();

    } catch (err) {
        options.onError(err instanceof Error ? err : new Error(String(err)));
    }
};

const wait = (milliseconds: number) => new Promise<void>(resolve => window.setTimeout(resolve, milliseconds));

const streamBackgroundReportAnalysis = async (
    file: File,
    scanType: VisionScanType,
    token: string,
    options: ChatServiceOptions,
) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('scan_type', scanType);

    const uploadResponse = await fetch(`${API_BASE_URL}/reports`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
    });
    if (!uploadResponse.ok) {
        throw new Error(`HTTP error! status: ${uploadResponse.status}`);
    }
    const job = await uploadResponse.json() as ReportUploadResponse;
    const stepId = genStepId();
    options.onStep({
        id: stepId,
        type: 'tool_start',
        tool: 'report_analysis',
        content: '正在后台分析报告…',
        isFinished: false,
    });

    const deadline = Date.now() + 5 * 60 * 1000;
    while (Date.now() < deadline) {
        const statusResponse = await fetch(`${API_BASE_URL}/reports/${job.report_id}`, {
            headers: { Authorization: `Bearer ${token}` },
        });
        if (!statusResponse.ok) {
            throw new Error(`HTTP error! status: ${statusResponse.status}`);
        }
        const status = await statusResponse.json() as ReportStatusResponse;
        if (status.analysis_status === 'completed') {
            options.onStepFinish('report_analysis');
            options.onChunk(status.analysis?.text ?? '报告识别已完成，但没有可展示的文本。');
            options.onDone();
            return;
        }
        if (status.analysis_status === 'failed' || status.analysis_status === 'queue_failed') {
            throw new Error(status.error ?? '报告分析失败');
        }
        await wait(1000);
    }
    throw new Error('报告分析超时，请稍后在报告列表中重试');
};
