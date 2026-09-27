import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { api } from './api';

/**
 * The backend mounts every API route under /api and the SPA owns /students,
 * /attendance and /dataset. These tests lock that contract in place: a
 * regression here silently breaks the deployed pages.
 *
 * Assertions match on the *path* rather than the full URL, because the
 * absolute origin legitimately changes with VITE_API_URL (separate host) while
 * the route shape must not.
 */
describe('api client', () => {
  let fetchMock;

  beforeEach(() => {
    fetchMock = vi.fn();
    globalThis.fetch = fetchMock;
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  const ok = (body) => ({ ok: true, status: 200, json: async () => body });
  const lastUrl = () => fetchMock.mock.calls.at(-1)[0];
  const pathOf = (url) => String(url).replace(/^https?:\/\/[^/]+/, '');

  it('prefixes every read endpoint with /api', async () => {
    fetchMock.mockResolvedValue(ok([]));
    await api.listStudents();
    expect(pathOf(lastUrl())).toBe('/api/students');
  });

  it('targets the documented health path', async () => {
    fetchMock.mockResolvedValue(ok({ status: 'healthy' }));
    await api.health();
    expect(pathOf(lastUrl())).toBe('/api/health');
  });

  it('does NOT call the un-prefixed data routes that collide with SPA pages', async () => {
    fetchMock.mockResolvedValue(ok([]));
    await api.listAttendance();
    await api.datasetStats();
    for (const call of fetchMock.mock.calls) {
      const p = pathOf(call[0]);
      expect(p).toMatch(/^\/api\//);
    }
  });

  it('sends a multipart body for file uploads instead of JSON', async () => {
    fetchMock.mockResolvedValue(ok({ status: 'unknown' }));
    const file = new File(['x'], 'a.jpg', { type: 'image/jpeg' });
    await api.predictFile(file);
    const [url, options] = fetchMock.mock.calls[0];
    expect(pathOf(url)).toBe('/api/predict');
    expect(options.body).toBeInstanceOf(FormData);
    expect(options.headers?.['Content-Type']).toBeUndefined();
  });

  it('encodes the student id in the path', async () => {
    fetchMock.mockResolvedValue(ok({}));
    await api.getStudent('S0001/../etc');
    expect(pathOf(lastUrl())).toBe('/api/students/S0001%2F..%2Fetc');
  });

  it('raises the backend detail message on failure', async () => {
    fetchMock.mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({ detail: 'Could not decode image' }),
    });
    await expect(api.predictFile(new File(['x'], 'a.jpg'))).rejects.toThrow(
      'Could not decode image',
    );
  });

  it('survives a non-JSON error body', async () => {
    fetchMock.mockResolvedValue({
      ok: false,
      status: 502,
      json: async () => {
        throw new Error('not json');
      },
    });
    await expect(api.health()).rejects.toThrow(/502/);
  });

  it('builds the CSV export url with only defined filters', () => {
    const url = api.exportUrl({ student_id: 'S0001', date: '', department: undefined });
    expect(pathOf(url)).toBe('/api/attendance/export.csv?student_id=S0001');
  });

  it('builds a date-filtered attendance query', async () => {
    fetchMock.mockResolvedValue(ok([]));
    await api.listAttendance({ date: '2026-01-05', student_id: '' });
    expect(pathOf(lastUrl())).toBe('/api/attendance?date=2026-01-05');
  });

  it('returns null for a 204 response', async () => {
    fetchMock.mockResolvedValue({ ok: true, status: 204, json: async () => ({}) });
    await expect(api.deleteStudent('S0001')).resolves.toBeNull();
  });
});
