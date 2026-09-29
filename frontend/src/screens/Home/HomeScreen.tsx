import { Activity, ChevronRight, CircleHelp, FileText, HeartPulse, LogOut, MessageSquare, Pill, Plus, ShieldCheck, Stethoscope } from 'lucide-react';
import { logout } from '../../services/chatService';
import { useGlobalStore } from '../../store/useGlobalStore';
import GlobalChatView from '../../components/chat/GlobalChatView';
import ChatModeHeader from '../../components/chat/ChatModeHeader';
import type { ChatMode } from '../../types';

const RAIL_ITEMS: { id: ChatMode; label: string; icon: typeof Stethoscope }[] = [
    { id: 'clinic', label: 'AI 辅助诊室', icon: Stethoscope },
    { id: 'report', label: '报告解读', icon: FileText },
    { id: 'pharmacy', label: '药品服务', icon: Pill },
    { id: 'insurance', label: '医保查询', icon: ShieldCheck },
    { id: 'dashboard', label: '健康概览', icon: Activity },
];

const HomeScreen = () => {
    const { isElderMode, setIsElderMode, chatMode, enterChatMode, messages, conversations, conversationId, selectConversation, createNewConversation, conversationError, setAccessToken } = useGlobalStore();
    const activeLabel = RAIL_ITEMS.find(item => item.id === chatMode)?.label ?? '综合健康咨询';

    return (
        <div className="workspace-main">
            <header className="topbar">
                <div className="brand-lockup">
                    <div className="brand-mark">DX</div>
                    <div>
                        <h1 className="brand-name">CareMate 健康工作台</h1>
                        <p className="brand-caption">安全优先的智能健康信息服务</p>
                    </div>
                </div>
                <div className="topbar-actions">
                    <div className="status-pill"><span className="status-dot" />服务正常</div>
                    <button className={`elder-toggle ${isElderMode ? 'active' : ''}`} onClick={() => setIsElderMode(!isElderMode)}>
                        {isElderMode ? '长辈模式已开' : '长辈模式'}
                    </button>
                    <button className="icon-button" aria-label="退出登录" title="退出登录" onClick={() => void logout().finally(() => setAccessToken(null))}><LogOut size={17} /></button>
                    <button className="icon-button" aria-label="帮助"><CircleHelp size={17} /></button>
                </div>
            </header>

            <div className="workspace-grid">
                <aside className="panel context-panel">
                    <p className="panel-label">当前工作区</p>
                    <div className="mode-card">
                        <div className="mode-label"><span className="mode-label-icon"><HeartPulse size={16} /></span>{activeLabel}</div>
                        <p className="mode-description">问题会被拆分为安全分流、资料检索和专业回答等步骤，结果经过来源与风险检查。</p>
                    </div>
                    <nav className="rail-list" aria-label="健康服务">
                        {RAIL_ITEMS.map(item => {
                            const Icon = item.icon;
                            return <button key={item.id} className={`rail-item ${chatMode === item.id ? 'active' : ''}`} onClick={() => enterChatMode(item.id)}><Icon size={16} /><span>{item.label}</span><ChevronRight size={13} className="ml-auto" /></button>;
                        })}
                    </nav>
                    <div className="conversation-list-section">
                        <div className="conversation-list-heading">
                            <span>会话</span>
                            <button type="button" className="icon-button" onClick={() => void createNewConversation()} title="新建会话" aria-label="新建会话"><Plus size={16} /></button>
                        </div>
                        {conversationError && <p className="conversation-error" role="status">{conversationError}</p>}
                        <div className="conversation-list" aria-label="会话列表">
                            {conversations.slice(0, 6).map(conversation => (
                                <button
                                    key={conversation.id}
                                    className={`conversation-list-item ${conversation.id === conversationId ? 'active' : ''}`}
                                    onClick={() => void selectConversation(conversation.id)}
                                    title={conversation.title}
                                >
                                    <MessageSquare size={14} />
                                    <span>{conversation.title || '新对话'}</span>
                                </button>
                            ))}
                        </div>
                    </div>
                    <div className="rail-footnote">急症信号由确定性安全规则优先识别。系统不能替代医生诊断或急诊处置。</div>
                </aside>

                <section className="panel conversation-panel">
                    <div className="conversation-header">
                        <div>
                            <h2 className="conversation-title">{chatMode === 'general' ? '健康咨询' : activeLabel}</h2>
                            <p className="conversation-subtitle">基于对话、工具与医学资料的辅助判断</p>
                        </div>
                        <div className="conversation-header-actions">
                            <select className="mobile-conversation-select" value={conversationId ?? ''} onChange={event => void selectConversation(event.target.value)} aria-label="切换会话">
                                {conversations.map(conversation => <option key={conversation.id} value={conversation.id}>{conversation.title || '新对话'}</option>)}
                            </select>
                            <button type="button" className="mobile-new-conversation icon-button" onClick={() => void createNewConversation()} title="新建会话" aria-label="新建会话"><Plus size={16} /></button>
                            <div className="conversation-meta">{messages.length > 1 ? `${messages.length - 1} 条对话` : '新的咨询'}</div>
                        </div>
                    </div>
                    <ChatModeHeader />
                    <div className="chat-scroller"><GlobalChatView /></div>
                </section>

                <aside className="panel insight-panel">
                    <p className="panel-label">运行摘要</p>
                    <div className="insight-block">
                        <p className="insight-title">本次会话</p>
                        <div className="status-line"><span>会话状态</span><span className="status-value good">已连接</span></div>
                        <div className="status-line"><span>回答模型</span><span className="status-value">DeepSeek</span></div>
                    </div>
                    <div className="insight-block">
                        <p className="insight-title">安全边界</p>
                        <div className="status-line"><span>红旗检查</span><span className="status-value good">已启用</span></div>
                        <div className="status-line"><span>引用过滤</span><span className="status-value good">已启用</span></div>
                        <div className="status-line"><span>人工接管</span><span className="status-value">可用</span></div>
                    </div>
                    <div className="insight-block">
                        <div className="security-note"><strong>使用提示</strong>请提供症状持续时间、严重程度、既往病史和当前用药。涉及胸痛、呼吸困难、意识异常等情况，请立即联系急救服务。</div>
                    </div>
                </aside>
            </div>
        </div>
    );
};

export default HomeScreen;
