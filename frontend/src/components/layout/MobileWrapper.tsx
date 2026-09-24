import React from 'react';
import { useGlobalStore } from '../../store/useGlobalStore';

interface MobileWrapperProps {
  children: React.ReactNode;
}

const MobileWrapper: React.FC<MobileWrapperProps> = ({ children }) => {
  const { isElderMode } = useGlobalStore();

  return (
    <div className={`app-shell flex flex-col h-[100dvh] w-full relative font-sans overflow-hidden transition-all duration-300 ${isElderMode ? 'elder-mode' : ''}`}>
      {/* 渲染当前主屏幕内容 */}
      <main className="flex-1 overflow-hidden relative z-0 bg-slate-50">
        {children}
      </main>

      {/* 样式定义 */}
      <style dangerouslySetInnerHTML={{
        __html: `
        .scrollbar-hide::-webkit-scrollbar { display: none; }
        .scrollbar-hide { -ms-overflow-style: none; scrollbar-width: none; }
        
        @keyframes scan {
          0% { top: 0; }
          100% { top: 100%; }
        }
        .animate-scan {
          animation: scan 2s linear infinite;
        }

        /* 长辈模式全局字体放大与加粗 */
        .elder-mode h1, .elder-mode h2, .elder-mode h3, .elder-mode p, .elder-mode input, .elder-mode span, .elder-mode div {
          letter-spacing: 0.03em;
        }
        .elder-mode .message-bubble, .elder-mode .message-input { font-size: 1.08rem !important; line-height: 1.7; }
        .elder-mode .suggestion-chip, .elder-mode .dock-nav-item { font-size: .9rem !important; }
      `}} />
    </div>
  );
};

export default MobileWrapper;
