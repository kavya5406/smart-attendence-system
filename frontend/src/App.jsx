import { Route, Routes } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import FaceRecognition from './pages/FaceRecognition';
import Students from './pages/Students';
import Dataset from './pages/Dataset';
import Attendance from './pages/Attendance';
import Reports from './pages/Reports';

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/recognition" element={<FaceRecognition />} />
        <Route path="/students" element={<Students />} />
        <Route path="/dataset" element={<Dataset />} />
        <Route path="/attendance" element={<Attendance />} />
        <Route path="/reports" element={<Reports />} />
        <Route path="*" element={<Dashboard />} />
      </Routes>
    </Layout>
  );
}
