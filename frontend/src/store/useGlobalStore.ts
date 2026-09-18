import { useContext } from 'react';
import { GlobalContext } from './GlobalContext';

export const useGlobalStore = () => {
    const context = useContext(GlobalContext);
    if (context === undefined) {
        throw new Error('useGlobalStore must be used within a GlobalProvider');
    }
    return context;
};
