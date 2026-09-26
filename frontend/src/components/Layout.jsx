import { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import api from '../api/api';

const LINKS = [
  { to: '/', label: 'Dashboard', icon: '🏠', end: true },
  { to: '/recognition', label: 'Face Recognition', icon: '📸' },
  { to: '/students', label: 'Students', icon: '👥' },
  { to: '/dataset', label: 'Dataset', icon: '📁' },
  { to: '/attendance', label: 'Attendance', icon: '✅' },
  { to: '/reports', label: 'Reports', icon: '📊' },
];

export default function Layout({ children }) {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ status: 'offline' }));
  }, []);

  const up = health?.status === 'healthy';
  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="logo">📸</span>
          <h1>Smart Attendance</h1>
        </div>
        <p className="brand-sub">Face recognition system</p>

        <nav className="nav">
          {LINKS.map((l) => (
            <NavLink key={l.to} to={l.to} end={l.end} className={({ isActive }) => (isActive ? 'active' : '')}>
              <span className="icon">{l.icon}</span>
              {l.label}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div><span className={`status-dot ${up ? 'up' : 'down'}`} />{up ? 'System healthy' : 'System offline'}</div>
          {health?.model_type && <div style={{ marginTop: 4 }}>Model: {health.model_type}</div>}
          {health?.known_students && (
            <div>Identities: {health.known_students.length}</div>
          )}
        </div>
      </aside>

      <main className="main">{children}</main>
    </div>
  );
}
