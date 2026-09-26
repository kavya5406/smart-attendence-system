/**
 * API client.
 *
 * Requests are relative by default so the frontend works unchanged when it is
 * served by the FastAPI backend (single origin) and when it runs on the Vite
 * dev server (which proxies the same paths). Set VITE_API_URL to point at a
 * separately hosted backend.
 */

const BASE = `${(import.meta.env.VITE_API_URL || '').replace(/\/$/, '')}/api`;

async function request(path, options = {}) {
  const response = await fetch(`${BASE}${path}`, options);
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      /* non-JSON error body */
    }
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  if (response.status === 204) return null;
  return response.json();
}

const json = (method, body) => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request('/health'),
  modelInfo: () => request('/model/info'),

  // ---- prediction ----
  async predictBase64(dataUrl) {
    return request('/predict-base64', json('POST', { image_data: dataUrl }));
  },
  async predictFile(file) {
    const form = new FormData();
    form.append('file', file);
    return request('/predict', { method: 'POST', body: form });
  },

  // ---- students ----
  listStudents: () => request('/students'),
  getStudent: (id) => request(`/students/${encodeURIComponent(id)}`),
  createStudent: (payload) => request('/students', json('POST', payload)),
  updateStudent: (id, payload) =>
    request(`/students/${encodeURIComponent(id)}`, json('PATCH', payload)),
  deleteStudent: (id, removeImages = false) =>
    request(`/students/${encodeURIComponent(id)}?remove_images=${removeImages}`, { method: 'DELETE' }),

  // ---- dataset ----
  datasetStats: (deep = false, checkFaces = false) =>
    request(`/dataset/stats?deep=${deep}&check_faces=${checkFaces}`),
  async uploadZip(file) {
    const form = new FormData();
    form.append('file', file);
    return request('/students/upload-zip', { method: 'POST', body: form });
  },
  async uploadStudentImages(studentId, name, files, department = '', year = '') {
    const form = new FormData();
    Array.from(files).forEach((f) => form.append('files', f));
    form.append('student_id', studentId);
    form.append('student_name', name);
    form.append('department', department);
    form.append('year', year);
    return request('/students/upload', { method: 'POST', body: form });
  },
  rebuildVerifier: () => request('/dataset/rebuild-verifier', { method: 'POST' }),

  // ---- attendance ----
  markAttendance: (payload) => request('/attendance', json('POST', payload)),
  listAttendance: (params = {}) => {
    const query = new URLSearchParams(
      Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== '')
    ).toString();
    return request(`/attendance${query ? `?${query}` : ''}`);
  },
  todayAttendance: () => request('/attendance/today'),
  dashboardStats: () => request('/dashboard/stats'),
  predictionStats: () => request('/stats/predictions'),
  exportUrl: (params = {}) => {
    const query = new URLSearchParams(
      Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== '')
    ).toString();
    return `${BASE}/attendance/export.csv${query ? `?${query}` : ''}`;
  },
};

export default api;
