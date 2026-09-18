import type { FC } from 'react';
import { CreditCard, History, User, Zap } from 'lucide-react';

const SERVICES = [
    { label: '消费记录', sub: '查询医保消费记录', icon: History, color: 'text-teal-500', bg: 'bg-teal-50' },
    { label: '年度缴费', sub: '查询医保缴费明细', icon: CreditCard, color: 'text-blue-500', bg: 'bg-blue-50' },
    { label: '共济管理', sub: '管理家庭共济关系', icon: User, color: 'text-indigo-500', bg: 'bg-indigo-50' },
    { label: '异地备案', sub: '查询跨省就医备案', icon: Zap, color: 'text-orange-500', bg: 'bg-orange-50' },
];

const ServiceGrid: FC = () => {
    return (
        <div className="grid grid-cols-2 gap-4">
            {SERVICES.map((item, i) => (
                <div key={i} className="bg-white p-5 rounded-3xl border border-slate-100 cursor-pointer hover:border-blue-200 transition-colors shadow-sm">
                    <div className={`w-10 h-10 rounded-full ${item.bg} flex items-center justify-center mb-3`}>
                        <item.icon size={20} className={item.color} />
                    </div>
                    <div className="font-bold mb-1 tracking-wide text-slate-800">{item.label}</div>
                    <div className="text-xs text-slate-500 line-clamp-1">{item.sub}</div>
                </div>
            ))}
        </div>
    );
};

export default ServiceGrid;
