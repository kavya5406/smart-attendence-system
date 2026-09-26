import { useEffect, useState } from 'react';
import api from '../api/api';

export default function Students() {
  const [students, setStudents] = useState(null);
  const [error, setError] = useState('');
  const [form, setForm] = useState({ student_id: '', student_name: '', department: '', year: '', email: '' });
  const [editing, setEditing] = useState(null);

  const load = () => api.listStudents().then(setStudents).catch((e) => setError(e.message));
  useEffect(() => { load(); }, []);

  const submit = async (event) => {
    event.preventDefault();
    setError('');
    try {
      if (editing) {
        await api.updateStudent(editing, form);
        setEditing(null);
      } else {
        await api.createStudent(form);
      }
      setForm({ student_id: '', student_name: '', department: '', year: '', email: '' });
      load();
    } catch (e) { setError(e.message); }
  };

  const remove = async (id) => {
    if (!window.confirm(`Delete ${id}? Attendance history for this student is also removed.`)) return;
    try { await api.deleteStudent(id); load(); } catch (e) { setError(e.message); }
  };

  const startEdit = (s) => {
    setEditing(s.student_id);
    setForm({ student_id: s.student_id, student_name: s.student_name, department: s.department, year: s.year, email: s.email });
  };

  if (error && !students) return <div className="alert error">{error}</div>;

  return (
    <div>
      <div className="page-head"><h2>Students</h2><p>Every student registered in the database.</p></div>
      {error && <div className="alert error">{error}</div>}

      <div className="card mb">
        <h3>{editing ? `Edit ${editing}` : 'Add student'}</h3>
        <form onSubmit={submit}>
          <div className="form-row">
            <label className="field">
              <span>Student ID</span>
              <input required disabled={!!editing} value={form.student_id}
                onChange={(e) => setForm({ ...form, student_id: e.target.value })} placeholder="S0001" />
            </label>
            <label className="field">
              <span>Student name</span>
              <input required value={form.student_name}
                onChange={(e) => setForm({ ...form, student_name: e.target.value })} placeholder="Lakshmi P" />
            </label>
            <label className="field">
              <span>Department</span>
              <input value={form.department} onChange={(e) => setForm({ ...form, department: e.target.value })} />
            </label>
            <label className="field">
              <span>Year</span>
              <input value={form.year} onChange={(e) => setForm({ ...form, year: e.target.value })} />
            </label>
            <label className="field">
              <span>Email</span>
              <input type="email" value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} />
            </label>
          </div>
          <div className="row">
            <button className="btn" type="submit">{editing ? 'Save changes' : '➕ Add student'}</button>
            {editing && <button className="btn secondary" type="button" onClick={() => { setEditing(null); setForm({ student_id: '', student_name: '', department: '', year: '', email: '' }); }}>Cancel</button>}
          </div>
        </form>
      </div>

      <div className="card">
        <h3>All students ({students?.length ?? 0})</h3>
        {!students ? <div className="empty">Loading…</div>
          : students.length === 0 ? <div className="empty">No students registered yet.</div> : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Student ID</th><th>Name</th><th>Department</th><th>Year</th>
                  <th>Images</th><th>Registered</th><th className="right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {students.map((s) => (
                  <tr key={s.student_id}>
                    <td className="mono">{s.student_id}</td>
                    <td>{s.student_name}</td>
                    <td>{s.department || <span className="muted">—</span>}</td>
                    <td>{s.year || <span className="muted">—</span>}</td>
                    <td>{s.image_count}</td>
                    <td className="small muted">{(s.created_at || '').slice(0, 10)}</td>
                    <td className="right">
                      <div className="row" style={{ justifyContent: 'flex-end' }}>
                        <button className="btn sm secondary" onClick={() => startEdit(s)}>View / Edit</button>
                        <button className="btn sm danger" onClick={() => remove(s.student_id)}>Delete</button>
                      </div>
                    </td>
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
