import { useEffect, useMemo, useState } from 'react';
import api from '../api/api';

const today = () => new Date().toISOString().slice(0, 10);

export default function Attendance() {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState('');
  const [filters, setFilters] = useState({ start_date: today(), end_date: today(), student_id: '', status: '' });

  const load = () => {
    const params = {
      start_date: filters.start_date || undefined,
      end_date: filters.end_date || undefined,
      student_id: filters.student_id || undefined,
      status: filters.status || undefined,
    };
    api.listAttendance(params).then(setRows).catch((e) => setError(e.message));
  };

  useEffect(() => { load(); /* eslint-disable-next-line */ }, []);

  const exportCsv = () => {
    const params = {
      start_date: filters.start_date || undefined,
      end_date: filters.end_date || undefined,
      student_id: filters.student_id || undefined,
      status: filters.status || undefined,
    };
    window.open(api.exportUrl(params), '_blank');
  };

  const summary = useMemo(() => {
    const list = rows || [];
    const byStudent = new Map();
    list.forEach((r) => byStudent.set(r.student_id, (byStudent.get(r.student_id) || 0) + 1));
    return { total: list.length, students: byStudent.size, byStudent };
  }, [rows]);

  return (
    <div>
      <div className="page-head spread">
        <div>
          <h2>Attendance</h2>
          <p>History recorded from real face recognition events.</p>
        </div>
        <button className="btn secondary" onClick={exportCsv}>⬇ Export CSV</button>
      </div>

      {error && <div className="alert error">{error}</div>}

      <div className="card mb">
        <h3>Filters</h3>
        <div className="form-row">
          <label className="field"><span>From date</span>
            <input type="date" value={filters.start_date} onChange={(e) => setFilters({ ...filters, start_date: e.target.value })} />
          </label>
          <label className="field"><span>To date</span>
            <input type="date" value={filters.end_date} onChange={(e) => setFilters({ ...filters, end_date: e.target.value })} />
          </label>
          <label className="field"><span>Student ID</span>
            <input value={filters.student_id} onChange={(e) => setFilters({ ...filters, student_id: e.target.value })} placeholder="S0001" />
          </label>
          <label className="field"><span>Status</span>
            <select value={filters.status} onChange={(e) => setFilters({ ...filters, status: e.target.value })}>
              <option value="">All</option>
              <option value="present">Present</option>
              <option value="absent">Absent</option>
            </select>
          </label>
        </div>
        <div className="row">
          <button className="btn" onClick={load}>Apply filters</button>
          <button className="btn secondary" onClick={() => { setFilters({ start_date: '', end_date: '', student_id: '', status: '' }); setTimeout(load, 0); }}>Reset</button>
        </div>
      </div>

      <div className="grid grid-3 mb">
        <div className="card stat primary"><div className="label">Records</div><div className="value">{summary.total}</div></div>
        <div className="card stat"><div className="label">Distinct students</div><div className="value">{summary.students}</div></div>
        <div className="card stat success"><div className="label">Present</div><div className="value">{(rows || []).filter((r) => r.status === 'present').length}</div></div>
      </div>

      <div className="card">
        <h3>Records</h3>
        {!rows ? <div className="empty">Loading…</div>
          : rows.length === 0 ? <div className="empty">No attendance records match these filters.</div> : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Student ID</th><th>Student name</th><th>Date</th><th>Time</th><th>Status</th><th>Model confidence</th></tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.id ?? `${r.student_id}-${r.date}`}>
                    <td className="mono">{r.student_id}</td>
                    <td>{r.student_name}</td>
                    <td>{r.date}</td>
                    <td>{r.time}</td>
                    <td><span className={`badge ${r.status === 'present' ? 'ok' : 'warn'}`}>{r.status}</span></td>
                    <td className="small muted">{r.confidence != null ? r.confidence.toFixed(4) : '—'}</td>
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
