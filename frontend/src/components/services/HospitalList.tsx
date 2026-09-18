import { useEffect, useState, type FC } from 'react';
import { listHospitals, type HospitalItem } from '../../services/chatService';
import HospitalCard from './HospitalCard';

const HospitalList: FC = () => {
    const [hospitals, setHospitals] = useState<HospitalItem[]>([]);
    const [city, setCity] = useState('北京');
    const [error, setError] = useState(false);
    useEffect(() => { let active = true; listHospitals({ city }).then((items) => active && setHospitals(items)).catch(() => active && setError(true)); return () => { active = false; }; }, [city]);
    return (
        <>
            <div className="flex justify-between items-center mb-4 px-2">
                <h3 className="font-bold text-slate-800 text-lg">推荐医疗机构 ({city})</h3>
                <button type="button" className="text-xs text-blue-500 font-bold" onClick={() => setCity(city === '北京' ? '上海' : '北京')}>切换城市</button>
            </div>
            {error ? <div className="rounded-2xl border border-red-100 bg-red-50 p-6 text-center text-sm text-red-600">医疗机构暂时无法加载</div> : hospitals.length === 0 ? <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 p-6 text-center text-sm text-slate-500">暂无该城市的医疗机构</div> : <div className="space-y-3">{hospitals.map((hospital) => <HospitalCard key={hospital.id} hospital={{ ...hospital, tags: hospital.tags, distance: hospital.distance_km == null ? '距离未知' : `${hospital.distance_km}公里`, isAd: hospital.is_ad }} />)}</div>}
        </>
    );
};

export default HospitalList;
