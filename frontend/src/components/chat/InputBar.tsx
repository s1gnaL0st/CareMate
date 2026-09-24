import React, { useRef, useState } from 'react';
import { Barcode, FileText, Mic, PackageSearch, Plus, Send, X } from 'lucide-react';
import { useGlobalStore } from '../../store/useGlobalStore';
import type { VisionScanType } from '../../services/chatService';

interface AttachmentOption {
    scanType: VisionScanType;
    label: string;
    description: string;
    icon: React.ElementType;
}

const ATTACHMENT_OPTIONS: Record<VisionScanType, AttachmentOption> = {
    report: {
        scanType: 'report',
        label: '拍报告',
        description: '上传检查/检验单',
        icon: FileText,
    },
    drug_box: {
        scanType: 'drug_box',
        label: '拍药盒',
        description: '识别药品包装',
        icon: PackageSearch,
    },
    trace_code: {
        scanType: 'trace_code',
        label: '扫追溯码',
        description: '识别追溯信息',
        icon: Barcode,
    },
};

interface InputBarProps {
    inputValue: string;
    setInputValue: (val: string) => void;
    onSend: () => void;
    onVisionUpload: (file: File, scanType: VisionScanType) => void;
    isVisionUploading?: boolean;
}

const getAttachmentOrder = (chatMode: string): VisionScanType[] => {
    if (chatMode === 'report') return ['report', 'drug_box', 'trace_code'];
    if (chatMode === 'pharmacy') return ['drug_box', 'trace_code', 'report'];
    return ['report', 'drug_box', 'trace_code'];
};

const InputBar: React.FC<InputBarProps> = ({
    inputValue,
    setInputValue,
    onSend,
    onVisionUpload,
    isVisionUploading = false,
}) => {
    const { isElderMode, chatMode } = useGlobalStore();
    const [isAttachmentOpen, setIsAttachmentOpen] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const selectedScanTypeRef = useRef<VisionScanType>('report');

    const handlePickImage = (scanType: VisionScanType) => {
        selectedScanTypeRef.current = scanType;
        setIsAttachmentOpen(false);
        fileInputRef.current?.click();
    };

    const handleFileSelected = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        event.target.value = '';
        if (!file) return;
        onVisionUpload(file, selectedScanTypeRef.current);
    };

    const attachmentOptions = getAttachmentOrder(chatMode).map(scanType => ATTACHMENT_OPTIONS[scanType]);

    return (
        <div className="input-row">
            <input
                ref={fileInputRef}
                type="file"
                accept="image/png,image/jpeg,image/webp,application/pdf"
                capture="environment"
                className="hidden"
                onChange={handleFileSelected}
            />
            <button className="input-action" aria-label="语音输入">
                <Mic size={20} />
            </button>
            <div className="input-wrap">
                <input
                    type="text"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && onSend()}
                    placeholder={isElderMode ? "按住说话或发消息..." : "描述症状、问医保、查报告..."}
                    className="message-input"
                />
                <button
                    onClick={onSend}
                    className="send-button"
                >
                    <Send size={16} />
                </button>
            </div>
            <div className="relative">
                {isAttachmentOpen && (
                    <div className="absolute right-0 bottom-14 w-56 rounded-2xl border border-slate-100 bg-white shadow-xl shadow-slate-200/70 p-2 animate-in fade-in slide-in-from-bottom-2">
                        <div className="flex items-center justify-between px-2 py-1.5">
                            <span className="text-xs font-black text-slate-500">图片上传</span>
                            <button
                                type="button"
                                onClick={() => setIsAttachmentOpen(false)}
                                className="w-7 h-7 rounded-full flex items-center justify-center text-slate-400 hover:bg-slate-100 hover:text-slate-700"
                                aria-label="关闭附件菜单"
                            >
                                <X size={14} />
                            </button>
                        </div>
                        <div className="space-y-1">
                            {attachmentOptions.map(option => {
                                const Icon = option.icon;
                                return (
                                    <button
                                        key={option.scanType}
                                        type="button"
                                        onClick={() => handlePickImage(option.scanType)}
                                        disabled={isVisionUploading}
                                        className="w-full flex items-center gap-3 rounded-xl px-3 py-2.5 text-left hover:bg-teal-50 disabled:opacity-50 disabled:hover:bg-transparent transition-colors"
                                    >
                                        <span className="w-9 h-9 rounded-xl bg-teal-50 text-teal-600 flex items-center justify-center">
                                            <Icon size={18} />
                                        </span>
                                        <span className="min-w-0">
                                            <span className="block text-sm font-black text-slate-800">{option.label}</span>
                                            <span className="block text-xs text-slate-400">{option.description}</span>
                                        </span>
                                    </button>
                                );
                            })}
                        </div>
                    </div>
                )}
                <button
                    type="button"
                    onClick={() => setIsAttachmentOpen(v => !v)}
                    disabled={isVisionUploading}
                    className="input-action disabled:opacity-60"
                    aria-label="打开图片上传菜单"
                >
                    <Plus size={24} className={isAttachmentOpen ? 'rotate-45 transition-transform' : 'transition-transform'} />
                </button>
            </div>
        </div>
    );
};

export default InputBar;
