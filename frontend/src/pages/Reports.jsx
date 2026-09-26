import { useEffect, useState } from 'react';
import api from '../api/api';

/** Reports: recognition statistics and per-student attendance rates. */
const pct = (v) => (v == null ? 'n/a' : `${(v * 100).toFixed(1)}%`);

export default function Reports() {
  const [predictions, setPredictions] = useState(null);
  const [attendance, setAttendance] = useState(null);
  const [students, setStudents] = useState(null);
  const [model, setModel] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([
      api.predictionStats(),
      api.listAttendance({ limit: 10000 }),
      api.listStudents(),
      api.modelInfo(),
    ])
      .then(([p, a, s, m]) => { setPredictions(p); setAttendance(a); setStudents(s); setModel(m); })
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="alert error">{error}</div>;
  if (!predictions) return <div className="empty">Loading…</div>;

  const byStudent = {};
  (attendance || []).forEach((r) => {
    byStudent[r.student_id] = byStudent[r.student_id] || { name: r.student_name, days: new Set() };
    byStudent[r.student_id].days.add(r.date);
  });

  const total = predictions.total_predictions || 0;
  const known = model?.model?.known_students || [];
  const measured = model?.verification?.measured || {};

  return (
    <div>
      <div className="page-head">
        <h2>Reports</h2>
        <p>Recognition statistics from the prediction log and attendance totals from the database.</p>
      </div>

      <div className="grid grid-4 mb">
        <div className="card stat primary"><div className="label">Total predictions</div><div className="value">{total}</div></div>
        <div className="card stat success"><div className="label">Recognised</div><div className="value">{predictions.recognized}</div></div>
        <div className="card stat danger"><div className="label">Unknown</div><div className="value">{predictions.unknown}</div></div>
        <div className="card stat"><div className="label">Avg latency</div><div className="value" style={{ fontSize: 22 }}>{predictions.avg_latency_ms ? `${predictions.avg_latency_ms.toFixed(0)} ms` : 'n/a'}</div></div>
      </div>

      <div className="grid grid-2">
        <div className="card">
          <h3>Attendance days per student</h3>
          {Object.keys(byStudent).length === 0 ? <div className="empty">No attendance recorded yet.</div> : (
            <div className="table-wrap">
              <table>
                <thead><tr><th>Student ID</th><th>Name</th><th>Days present</th></tr></thead>
                <tbody>
                  {Object.entries(byStudent).map(([id, v]) => (
                    <tr key={id}>
                      <td className="mono">{id}</td>
                      <td>{v.name}</td>
                      <td>{v.days.size}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="card">
          <h3>Model &amp; verification</h3>
          {model?.model && (
            <div>
              <div className="kv"><span className="k">Model type</span><span className="v">{model.model.model_type}</span></div>
              <div className="kv"><span className="k">Raw feature dim</span><span className="v">{model.model.raw_feature_dim}</span></div>
              <div className="kv"><span className="k">Model input dim</span><span className="v">{model.model.model_feature_dim}</span></div>
              <div className="kv"><span className="k">Known students</span><span className="v">{model.model.known_student_count}</span></div>
              <div className="kv"><span className="k">Verification method</span><span className="v" style={{ fontSize: 11 }}>{model.verification?.method}</span></div>
              <div className="kv"><span className="k">Threshold</span><span className="v">{model.verification?.threshold}</span></div>
            </div>
          )}
          {Object.keys(measured).length > 0 ? (
            <div className="mt">
              <h4 style={{ margin: '14px 0 6px' }}>Measured on the registered dataset</h4>
              <div className="kv"><span className="k">Images measured</span><span className="v">{measured.registered_images}</span></div>
              <div className="kv"><span className="k">Classifier top-1 accuracy</span><span className="v">{pct(measured.classifier_top1_accuracy)}</span></div>
              <div className="kv"><span className="k">Verified identity accuracy</span><span className="v">{pct(measured.prototype_top1_accuracy)}</span></div>
              <div className="kv"><span className="k">Registered faces accepted</span><span className="v">{pct(measured.prototype_acceptance_rate_on_registered)}</span></div>
              <div className="kv"><span className="k">Unknown faces wrongly accepted</span><span className="v">{pct(measured.false_acceptance_rate_on_negatives)}</span></div>
              <div className="kv"><span className="k">Separation (AUC)</span><span className="v">{measured.similarity_auc_registered_vs_negative ?? 'n/a'}</span></div>
              <p className="small muted mt">
                Measured by <code>scripts/measure_inference.py</code> against{' '}
                {measured.negative_samples} noise/blank samples. These are measured values, not estimates.
              </p>
            </div>
          ) : (
            <p className="small muted mt">No measurement file found. Run <code>python3 scripts/measure_inference.py</code> to produce one.</p>
          )}
          <p className="small muted mt">{model?.verification?.note}</p>
          {model?.model?.training_metrics_note && (
            <p className="small muted">{model.model.training_metrics_note}</p>
          )}
          <p className="small muted mt">
            Identities the trained model can recognise: {known.length ? known.join(', ') : 'none'}.
          </p>
        </div>
      </div>
    </div>
  );
}
