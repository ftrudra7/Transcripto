import { Routes, Route } from 'react-router-dom';
import { AppShell } from '@/components/layout/AppShell';
import { Home } from '@/pages/Home';

function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<Home />} />
      </Route>
    </Routes>
  );
}

export default App;
