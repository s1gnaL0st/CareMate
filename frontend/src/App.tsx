

// Providers
import { GlobalProvider } from './store/GlobalContext';
import { useGlobalStore } from './store/useGlobalStore';
import './App.css';

// Layout
import MobileWrapper from './components/layout/MobileWrapper';
import BottomNav from './components/layout/BottomNav';
import ScannerOverlay from './components/layout/ScannerOverlay';

// Core chat components
import HomeScreen from './screens/Home/HomeScreen';

// Admin screens that still have standalone value (accessible from settings/links)
import DashboardScreen from './screens/Dashboard/DashboardScreen';
import AuthScreen from './screens/Auth/AuthScreen';


// The main Screen selector (only a few non-chat screens remain)
const ScreenRouter = () => {
  const { chatMode } = useGlobalStore();

  // Dashboard is still accessible as a full immersive experience for health data
  if (chatMode === 'dashboard') {
    return <DashboardScreen />;
  }
  // Services booking page retains full-screen UX for its map/list experience
  // (but is now triggered from a chat card, not bottom nav)
  // Can be removed later once we have in-chat booking card

  // All other modes (clinic, insurance, pharmacy, report) now live inside the chat
  return <HomeScreen />;
};

const AppContent = () => {
  const { accessToken } = useGlobalStore();
  if (!accessToken) return <AuthScreen />;
  return (
    <MobileWrapper>
      <ScreenRouter />
      <BottomNav />
      <ScannerOverlay />
    </MobileWrapper>
  );
};

const App = () => {
  return (
    <GlobalProvider>
      <AppContent />
    </GlobalProvider>
  );
};

export default App;
