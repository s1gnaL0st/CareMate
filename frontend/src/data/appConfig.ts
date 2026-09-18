import { Camera, CreditCard, Heart, Activity, Stethoscope, Pill, type LucideIcon } from 'lucide-react';

export const SYSTEM_NAME = '大健康 AI';
export const DEFAULT_USER_LABEL = '用户';

export const MODULES: { id: string; name: string; icon: LucideIcon }[] = [
    { id: 'clinic', name: 'AI诊室', icon: Stethoscope },
    { id: 'dashboard', name: '健康小目标', icon: Heart },
    { id: 'pharmacy', name: '药管家', icon: Pill },
    { id: 'insurance', name: '查医保', icon: CreditCard },
    { id: 'report', name: '拍报告', icon: Camera },
    { id: 'services', name: '就医服务', icon: Activity },
];

export const SUGGESTIONS = [
    '想开点中药调理身体挂什么科？',
    '我最近总是失眠怎么办？',
    '帮我查一下医保余额',
    '扫码查一下这个感冒药真伪',
];

export interface DashboardHabitView {
    title: string;
    current: string;
    progress: number;
    color: string;
}

export interface HospitalListingView {
    name: string;
    tags: string[];
    distance: string;
    isAd: boolean;
}
