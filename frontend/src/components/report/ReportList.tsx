import { useEffect, useState, type FC } from 'react';
import { listReports, type ReportListItem } from '../../services/chatService';

const ReportList: FC = () => {
    const [reports, setReports] = useState<ReportListItem[]>([]);
    const [error, setError] = useState(false);
    useEffect(() => {
        let active = true;
        listReports().then((items) => active && setReports(items)).catch(() => active && setError(true));
        return () => { active = false; };
    }, []);
    return (
        <>
            <div className="flex justify-between items-center mb-4 px-2">
                <h3 className="font-bold text-slate-800 text-lg">历史报告记录</h3>
                <span className="text-xs text-slate-500 font-normal cursor-pointer bg-slate-200 px-2 py-1 rounded-md">按时间排序 ▼</span>
            </div>
            {error ? <div className="rounded-2xl border border-red-100 bg-red-50 p-6 text-center text-sm text-red-600">报告记录暂时无法加载，请稍后重试</div> : reports.length === 0 ? <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 p-6 text-center text-sm text-slate-500">上传检查报告后将在这里显示分析记录</div> : <div className="space-y-2">{reports.map((report) => <div key={report.report_id} className="flex items-center justify-between rounded-xl border border-slate-100 bg-white px-4 py-3"><div><div className="font-medium text-slate-800">{report.original_filename}</div><div className="text-xs text-slate-500">{new Date(report.created_at).toLocaleString()}</div></div><span className="text-xs text-slate-500">{report.analysis_status || report.status}</span></div>)}</div>}
        </>
    );
};

export default ReportList;
