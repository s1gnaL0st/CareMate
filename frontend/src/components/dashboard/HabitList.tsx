import { useEffect, useState, type FC } from 'react';
import { listHabitGoals, updateHabitGoal, type HabitGoal } from '../../services/chatService';
import HabitRow from './HabitRow';

const HabitList: FC = () => {
    const [goals, setGoals] = useState<HabitGoal[]>([]);
    useEffect(() => { let active = true; listHabitGoals().then((items) => active && setGoals(items)).catch(() => undefined); return () => { active = false; }; }, []);
    const toggle = async (goal: HabitGoal) => { const current = goal.current_value >= goal.target_value ? 0 : goal.current_value + 1; try { const updated = await updateHabitGoal(goal.id, current); setGoals((items) => items.map((item) => item.id === updated.id ? updated : item)); } catch { /* keep current state */ } };
    return (
        <div>
            <h3 className="font-bold text-slate-800 mb-4 px-2 text-lg">每日打卡计划</h3>
            {goals.length === 0 ? <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 p-6 text-center text-sm text-slate-500">创建健康目标后将在这里显示</div> : <div className="space-y-3">{goals.map((goal) => <button type="button" className="block w-full text-left" key={goal.id} onClick={() => void toggle(goal)}><HabitRow task={{ title: goal.title, current: `${goal.current_value}/${goal.target_value}${goal.unit}`, progress: Math.min(100, Math.round(goal.current_value / goal.target_value * 100)), color: goal.color }} /></button>)}</div>}
        </div>
    );
};

export default HabitList;
