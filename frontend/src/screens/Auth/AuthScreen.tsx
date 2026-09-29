import { useState } from 'react';
import type { FormEvent } from 'react';
import { HeartPulse, LockKeyhole, Mail, UserRound } from 'lucide-react';
import { login, register } from '../../services/chatService';
import { useGlobalStore } from '../../store/useGlobalStore';

const AuthScreen = () => {
    const { setAccessToken } = useGlobalStore();
    const [isRegister, setIsRegister] = useState(false);
    const [email, setEmail] = useState('admin');
    const [password, setPassword] = useState('123');
    const [name, setName] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const submit = async (event: FormEvent) => {
        event.preventDefault();
        setError(''); setLoading(true);
        try {
            const result = isRegister
                ? await register(email, password, name || '用户')
                : await login(email, password);
            setAccessToken(result.access_token);
        } catch {
            setError(isRegister ? '注册失败，请检查邮箱和密码。' : '账号或密码错误，请重试。');
        } finally { setLoading(false); }
    };

    return <main className="auth-shell">
        <section className="auth-card">
            <div className="auth-logo"><HeartPulse size={24} /></div>
            <p className="auth-kicker">CAREMATE HEALTH WORKSPACE</p>
            <h1>{isRegister ? '创建 CareMate 账号' : '登录 CareMate'}</h1>
            <p className="auth-subtitle">安全优先的智能健康信息服务</p>
            <form onSubmit={submit} className="auth-form">
                {isRegister && <label><UserRound size={16} /><input value={name} onChange={e => setName(e.target.value)} placeholder="姓名" /></label>}
                <label><Mail size={16} /><input value={email} onChange={e => setEmail(e.target.value)} placeholder="邮箱或 admin" required /></label>
                <label><LockKeyhole size={16} /><input type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder="密码" required /></label>
                {error && <p className="auth-error" role="alert">{error}</p>}
                <button className="auth-submit" disabled={loading}>{loading ? '处理中…' : isRegister ? '注册并进入' : '登录'}</button>
            </form>
            {!isRegister && <p className="auth-demo">开发环境默认账号：<b>admin</b> / <b>123</b></p>}
            <button className="auth-switch" onClick={() => { setIsRegister(v => !v); setError(''); }}>{isRegister ? '已有账号？返回登录' : '还没有账号？立即注册'}</button>
        </section>
    </main>;
};

export default AuthScreen;
