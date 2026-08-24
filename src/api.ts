import type {
  Capabilities,
  Project,
  RecognitionDiagnostics,
  RecognitionEvent,
  RecognitionMeasure,
  SyncPoint,
  User,
  VideoAnalysisRequest
} from "./types";

type ErrorPayload = { detail?: string };

async function request<T>(url: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(url, {
    credentials: "include",
    ...options,
    headers: {
      ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...options.headers
    }
  });

  if (!response.ok) {
    let message = `请求失败（${response.status}）`;
    try {
      const payload = (await response.json()) as ErrorPayload;
      if (payload.detail) message = payload.detail;
    } catch {
      // Keep the status-based message for non-JSON errors.
    }
    throw new Error(message);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  health: () => request<{ status: string; capabilities: Capabilities }>("/api/health"),
  me: () => request<User>("/api/auth/me"),
  register: (payload: { email: string; display_name: string; password: string }) =>
    request<User>("/api/auth/register", { method: "POST", body: JSON.stringify(payload) }),
  login: (payload: { email: string; password: string }) =>
    request<User>("/api/auth/login", { method: "POST", body: JSON.stringify(payload) }),
  logout: () => request<void>("/api/auth/logout", { method: "POST" }),
  projects: () => request<Project[]>("/api/projects"),
  project: (id: string) => request<Project>(`/api/projects/${id}`),
  createProject: (payload: { source_input: string; title: string; rights_confirmed: boolean }) =>
    request<Project>("/api/projects", { method: "POST", body: JSON.stringify(payload) }),
  createManualTabProject: (payload: { title: string; measure_count: number; tempo_bpm: number }) =>
    request<Project>("/api/projects/manual-tab", { method: "POST", body: JSON.stringify(payload) }),
  renameProject: (id: string, title: string) =>
    request<Project>(`/api/projects/${id}`, { method: "PATCH", body: JSON.stringify({ title }) }),
  updateProjectRights: (id: string, rightsConfirmed: boolean) =>
    request<Project>(`/api/projects/${id}/rights`, {
      method: "PATCH",
      body: JSON.stringify({ rights_confirmed: rightsConfirmed })
    }),
  lockProjectTempo: (id: string, tempoBpm: number) =>
    request<Project>(`/api/projects/${id}/tempo`, {
      method: "PATCH",
      body: JSON.stringify({ tempo_bpm: tempoBpm })
    }),
  inspectProject: (id: string) => request<Project>(`/api/projects/${id}/inspect`, { method: "POST" }),
  downloadProject: (id: string) =>
    request<{ status: string; message: string }>(`/api/projects/${id}/download`, { method: "POST" }),
  analyzeVideo: (id: string, payload: VideoAnalysisRequest) =>
    request<{ status: string; message: string; estimated_frames: number; source_fps: number }>(
      `/api/projects/${id}/video-analysis`,
      { method: "POST", body: JSON.stringify(payload) }
    ),
  recognizeProject: (id: string) =>
    request<{ status: string; message: string; engine: "tablature" | "staff" }>(`/api/projects/${id}/recognize`, { method: "POST" }),
  recognition: (id: string) => request<RecognitionDiagnostics>(`/api/projects/${id}/recognition`),
  retryRecognitionMeasure: (id: string, measure: number) =>
    request<RecognitionMeasure & { source_frame: number; source_name: string }>(
      `/api/projects/${id}/recognition/measures/${measure}/retry`,
      { method: "POST" }
    ),
  updateRecognitionMeasure: (id: string, measure: number, events: RecognitionEvent[], timeSignature?: { numerator: number; denominator: number }) =>
    request<RecognitionDiagnostics>(`/api/projects/${id}/recognition/measures/${measure}`, {
      method: "PATCH",
      body: JSON.stringify({ events, ...(timeSignature ? { time_signature: timeSignature } : {}) })
    }),
  appendRecognitionMeasure: (id: string, afterMeasure?: number) =>
    request<RecognitionDiagnostics>(
      `/api/projects/${id}/recognition/measures${afterMeasure === undefined ? "" : `?after_measure=${afterMeasure}`}`,
      { method: "POST" }
    ),
  uploadImages: (id: string, files: FileList | File[]) => {
    const body = new FormData();
    Array.from(files).forEach((file) => body.append("files", file));
    return request<Project>(`/api/projects/${id}/score-images`, { method: "POST", body });
  },
  uploadPdf: (id: string, file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<Project>(`/api/projects/${id}/score-pdf`, { method: "POST", body });
  },
  uploadAudio: (id: string, file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<Project>(`/api/projects/${id}/audio`, { method: "POST", body });
  },
  analyzeAudio: (id: string, source: "auto" | "uploaded" | "video" = "auto") =>
    request<{ status: string; message: string; source: "uploaded_audio" | "video_audio" }>(
      `/api/projects/${id}/audio-analysis?source=${source}`,
      { method: "POST" }
    ),
  applyAudioAlignment: (id: string) =>
    request<Project>(`/api/projects/${id}/audio-analysis/apply`, { method: "POST" }),
  uploadScore: (id: string, file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<Project>(`/api/projects/${id}/score-file`, { method: "POST", body });
  },
  addSyncPoint: (
    id: string,
    payload: { measure_number: number; time_seconds: number; score_position: number; label: string }
  ) =>
    request<SyncPoint>(`/api/projects/${id}/sync-points`, {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  deleteSyncPoint: (projectId: string, pointId: number) =>
    request<void>(`/api/projects/${projectId}/sync-points/${pointId}`, { method: "DELETE" })
};
