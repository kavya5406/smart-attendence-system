import { useCallback, useEffect, useRef, useState } from 'react';
import api from '../api/api';

/** Student dataset management: upload, add, view, remove and real statistics. */
export default function Dataset() {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const [deep, setDeep] = useState(false);

  // single-student form
  const [form, setForm] = useState({ student_id: '', student_name: '', department: '', year: '' });
  const [files, setFiles] = useState([]);
  const zipRef = useRef(null);
  const imgRef = useRef(null);

  const load = useCallback(() => {
    api.datasetStats(deep).then((s) => { setStats(s); setError(''); }).catch((e) => setError(e.message));
  }, [deep]);

  useEffect(() => { load(); }, [load]);

  const refreshAfter = async (msg) => {
    setMessage(msg);
    setBusy(false);
    // Prototypes are rebuilt server-side after a dataset change.
    await api.rebuildVerifier().catch(() => {});
    load();
  };

  const handleZip = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setBusy(true); setError(''); setMessage('');
    try {
      const res = await api.uploadZip(file);
      await refreshAfter(
        `Imported ${res.images_imported} images for ${res.students_imported} students` +
        (res.skipped_entries ? ` (${res.skipped_entries} metadata entries skipped)` : '')
      );
    } catch (e) { setError(e.message); setBusy(false); }
    event.target.value = '';
  };

  const handleImages = async (event) => {
    event.preventDefault();
    if (!form.student_id.trim() || !form.student_name.trim()) {
      setError('Student ID and name are both required.');
      return;
    }
    if (files.length === 0) {
      setError('Choose at least one image.');
      return;
    }
    setBusy(true); setError(''); setMessage('');
    try {
      const res = await api.uploadStudentImages(
        form.student_id.trim(), form.student_name.trim(), files, form.department, form.year
      );
      setForm({ student_id: '', student_name: '', department: '', year: '' });
      setFiles([]);
      if (imgRef.current) imgRef.current.value = '';
      await refreshAfter(`Saved ${res.images_saved} images for ${res.student_id}`);
    } catch (e) { setError(e.message); setBusy(false); }
  };

  const removeStudent = async (studentId) => {
    if (!window.confirm(`Remove ${studentId} and all of its images? This cannot be undone.`)) return;
    setBusy(true); setError(''); setMessage('');
    try {
      await api.deleteStudent(studentId, true);
      await refreshAfter(`Removed ${studentId}`);
    } catch (e) { setError(e.message); setBusy(false); }
  };

  if (error && !stats) return <div className="alert error">{error}</div>;
  if (!stats) return <div className="empty">Loading dataset…</div>;

  return (
    <div>
      <div className="page-head">
        <h2>Student Dataset</h2>
        <p>
          Register face images so the model can recognise students. Layout expected in a ZIP:
          one folder per student (<code>STUDENT_001/</code>) containing JPG, JPEG or PNG images.
          macOS metadata (<code>__MACOSX</code>, <code>.DS_Store</code>, <code>._*</code>) is ignored automatically.
        </p>
      </div>

      {error && <div className="alert error">{error}</div>}
      {message && <div className="alert success">{message}</div>}

      <div className="grid grid-4 mb">
        <div className="card stat primary">
          <div className="label">Total Students</div><div className="value">{stats.total_students}</div>
        </div>
        <div className="card stat">
          <div className="label">Total Images</div><div className="value">{stats.total_images}</div>
        </div>
        <div className="card stat success">
          <div className="label">Valid Images</div><div className="value">{stats.valid_images}</div>
        </div>
        <div className="card stat danger">
          <div className="label">Invalid Images</div><div className="value">{stats.invalid_images}</div>
          <div className="sub">+ {stats.duplicate_images} duplicate</div>
        </div>
      </div>

      <div className="grid grid-2">
        <div className="card">
          <h3>Upload dataset ZIP</h3>
          <p className="small muted">
            One folder per student. The trained model can only recognise the identities it was
            trained on (<code>S0001</code>–<code>S0005</code>); new students need a retrain.
          </p>
          <button className="btn" disabled={busy} onClick={() => zipRef.current?.click()}>
            {busy ? 'Working…' : '📦 Upload dataset ZIP'}
          </button>
          <input ref={zipRef} type="file" accept=".zip" hidden onChange={handleZip} />
          <p className="small muted mt">
            Dataset path: <code>{stats.dataset_path}</code>
          </p>
        </div>

        <div className="card">
          <h3>Add student</h3>
          <form onSubmit={handleImages}>
            <div className="form-row">
              <label className="field">
                <span>Student ID</span>
                <input value={form.student_id} onChange={(e) => setForm({ ...form, student_id: e.target.value })} placeholder="S0001" />
              </label>
              <label className="field">
                <span>Student name</span>
                <input value={form.student_name} onChange={(e) => setForm({ ...form, student_name: e.target.value })} placeholder="Lakshmi P" />
              </label>
              <label className="field">
                <span>Department</span>
                <input value={form.department} onChange={(e) => setForm({ ...form, department: e.target.value })} placeholder="CSE" />
              </label>
              <label className="field">
                <span>Year</span>
                <input value={form.year} onChange={(e) => setForm({ ...form, year: e.target.value })} placeholder="3" />
              </label>
            </div>
            <label className="field">
              <span>Face images (JPG / JPEG / PNG)</span>
              <input ref={imgRef} type="file" multiple accept="image/*" onChange={(e) => setFiles(Array.from(e.target.files || []))} />
            </label>
            {files.length > 0 && <p className="small muted">{files.length} file(s) selected</p>}
            <button className="btn" disabled={busy} type="submit">
              {busy ? 'Saving…' : '➕ Add student'}
            </button>
          </form>
        </div>
      </div>

      <div className="card mt">
        <div className="spread">
          <h3 style={{ margin: 0 }}>Registered identities</h3>
          <label className="row small muted" style={{ cursor: 'pointer' }}>
            <input type="checkbox" checked={deep} onChange={(e) => setDeep(e.target.checked)} style={{ width: 'auto' }} />
            Deep scan (decode every image)
          </label>
        </div>

        {stats.students.length === 0 ? (
          <div className="empty">No student folders found in the dataset directory.</div>
        ) : (
          <div className="table-wrap mt">
            <table>
              <thead>
                <tr>
                  <th>Student ID</th><th>Student name</th><th>Images</th>
                  <th>Valid</th><th>Invalid</th><th>Duplicates</th>
                  <th>Registration status</th><th className="right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {stats.students.map((s) => (
                  <tr key={s.student_id}>
                    <td className="mono">{s.student_id}</td>
                    <td>{s.student_name || <span className="muted">—</span>}</td>
                    <td>{s.image_count}</td>
                    <td>{s.valid_count}</td>
                    <td>{s.invalid_count > 0 ? <span className="badge bad">{s.invalid_count}</span> : 0}</td>
                    <td>{s.duplicate_count > 0 ? <span className="badge warn">{s.duplicate_count}</span> : 0}</td>
                    <td>
                      <span className={`badge ${s.registered ? 'ok' : 'info'}`}>
                        {s.registered ? 'registered' : 'not registered'}
                      </span>
                    </td>
                    <td className="right">
                      <button className="btn sm danger" disabled={busy} onClick={() => removeStudent(s.student_id)}>
                        Remove
                      </button>
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
