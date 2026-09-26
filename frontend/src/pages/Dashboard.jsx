import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import api from '../api/api';

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [error, setError] = useState('');

  const load = () => {
    Promise.all([api.dashboardStats(), api.datasetStats()])
      .then(([s, d]) => { setStats(s); setDataset(d); setError(''); })
      .catch((e) => setError(e.message || 'Could not load dashboard data.'));
  };

  useEffect(() => { load(); }, []);

  if (error) return <div className="alert error">{error}</div>;
  if (!stats) return <div className="empty">Loading…</div>;

  return (
    <div>
      <div className="card hero">
        <h2>Smart Attendance System</h2>
        <p>
          Face recognition attendance powered by an existing classical machine learning model
          (HOG + LBP features, LogisticRegression classifier). All figures below come from the
          live database and the registered dataset.
        </p>
        <div className="hero-actions">
          <Link className="btn success lg" to="/recognition">📸 Start Face Recognition</Link>
          <Link className="btn secondary lg" to="/dataset">📁 Upload Student Dataset</Link>
        </div>
      </div>

      <div className="grid grid-4 mb">
        <div className="card stat primary">
          <div className="label">Total Students</div>
          <div className="value">{stats.total_students}</div>
          <div className="sub">registered in database</div>
        </div>
        <div className="card stat success">
          <div className="label">Present Today</div>
          <div className="value">{stats.present_today}</div>
          <div className="sub">{stats.date}</div>
        </div>
        <div className="card stat danger">
          <div className="label">Absent Today</div>
          <div className="value">{stats.absent_today}</div>
          <div className="sub">not marked yet</div>
        </div>
        <div className="card stat">
          <div className="label">Attendance %</div>
          <div className="value">{stats.attendance_percentage}%</div>
          <div className="sub">for today</div>
        </div>
      </div>

      {dataset && (
        <div className="card mb">
          <div className="spread">
            <h3 style={{ margin: 0 }}>Dataset status</h3>
            <Link className="btn sm secondary" to="/dataset">Manage dataset</Link>
          </div>
          <div className="grid grid-4 mt">
            <div className="stat"><div className="label">Identities on disk</div><div className="value" style={{ fontSize: 24 }}>{dataset.total_students}</div></div>
            <div className="stat"><div className="label">Images</div><div className="value" style={{ fontSize: 24 }}>{dataset.total_images}</div></div>
            <div className="stat success"><div className="label">Valid</div><div className="value" style={{ fontSize: 24 }}>{dataset.valid_images}</div></div>
            <div className="stat warn"><div className="label">Invalid / duplicate</div><div className="value" style={{ fontSize: 24 }}>{dataset.invalid_images + dataset.duplicate_images}</div></div>
          </div>
        </div>
      )}

      <div className="card">
        <div className="spread">
          <h3 style={{ margin: 0 }}>Recent attendance</h3>
          <Link className="btn sm secondary" to="/attendance">View all</Link>
        </div>
        {stats.recent_attendance.length === 0 ? (
          <div className="empty">No attendance recorded yet. Recognise a student to get started.</div>
        ) : (
          <div className="table-wrap mt">
            <table>
              <thead>
                <tr>
                  <th>Student ID</th><th>Student Name</th><th>Date</th><th>Time</th><th>Status</th>
                </tr>
              </thead>
              <tbody>
                {stats.recent_attendance.map((row, i) => (
                  <tr key={row.id ?? i}>
                    <td className="mono">{row.student_id}</td>
                    <td>{row.student_name}</td>
                    <td>{row.date}</td>
                    <td>{row.time}</td>
                    <td><span className={`badge ${row.status === 'present' ? 'ok' : 'warn'}`}>{row.status}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
