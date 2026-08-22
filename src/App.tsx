import { FormEvent, lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowUpRight,
  AudioLines,
  BookOpen,
  Check,
  CheckCircle2,
  ChevronRight,
  Clock3,
  Download,
  FileMusic,
  FileText,
  FolderOpen,
  Headphones,
  ImagePlus,
  Library,
  Link2,
  LoaderCircle,
  LockKeyhole,
  LogOut,
  ListMusic,
  Music2,
  PencilLine,
  Plus,
  RefreshCw,
  Save,
  ScanLine,
  Scissors,
  Trash2,
  Upload,
  UserRound,
  Volume2,
  WandSparkles,
  Waves
} from "lucide-react";
import { api } from "./api";
import { ScoreReviewPanel } from "./components/ScoreReviewPanel";
import { VideoSliceEditor } from "./components/VideoSliceEditor";
import type { Capabilities, Project, RecognitionDiagnostics, RecognitionEvent, SyncPoint, User, VideoAnalysisRequest } from "./types";

const AlphaTabPlayer = lazy(() =>
  import("./components/AlphaTabPlayer").then((module) => ({ default: module.AlphaTabPlayer }))
);

const EMPTY_CAPABILITIES: Capabilities = {
  ffmpeg: false,
  yt_dlp: false,
  audiveris: false,
  tab_ocr: false,
  audio_analysis: false
};

const STATUS_LABELS: Record<string, string> = {
  draft: "草稿",
  downloading: "正在获取视频",
  video_ready: "视频已就绪",
  analyzing: "正在切片分析",
  frames_ready: "候选帧已生成",
  pdf_ready: "PDF 已生成",
  recognizing: "正在识别",
  score_ready: "乐谱可播放",
  failed: "需要处理"
};

function formatDate(value: string) {
  const date = new Date(value);
  return new Intl.DateTimeFormat("zh-CN", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(date);
}

function formatTime(seconds: number) {
  if (!Number.isFinite(seconds)) return "00:00.0";
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(1).padStart(4, "0")}`;
}

function routeProjectId() {
  const match = window.location.hash.match(/^#project=([a-f0-9]{32})$/);
  return match?.[1] ?? null;
}

function interpolateScorePosition(points: SyncPoint[], time: number) {
  if (!points.length) return null;
  const sorted = [...points].sort((a, b) => a.time_seconds - b.time_seconds);
  if (time <= sorted[0].time_seconds) return sorted[0].score_position;
  if (time >= sorted[sorted.length - 1].time_seconds) return sorted[sorted.length - 1].score_position;
  const nextIndex = sorted.findIndex((point) => point.time_seconds >= time);
  const previous = sorted[nextIndex - 1];
  const next = sorted[nextIndex];
  const span = next.time_seconds - previous.time_seconds;
  const progress = span <= 0 ? 0 : (time - previous.time_seconds) / span;
  return previous.score_position + (next.score_position - previous.score_position) * progress;
}

type Notice = { message: string; tone: "success" | "error" } | null;

function ProjectCover({ project }: { project: Project }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [project.cover_url]);

  return project.cover_url && !failed ? (
    <img
      src={project.cover_url}
      alt={`${project.title} 视频封面`}
      loading="lazy"
      referrerPolicy="no-referrer"
      onError={() => setFailed(true)}
    />
  ) : (
    <div className="cover-placeholder"><AudioLines size={30} /><span>{project.source_id}</span></div>
  );
}

function App() {
  const [user, setUser] = useState<User | null>(null);
  const [capabilities, setCapabilities] = useState<Capabilities>(EMPTY_CAPABILITIES);
  const [projectId, setProjectId] = useState<string | null>(routeProjectId());
  const [booting, setBooting] = useState(true);
  const [notice, setNotice] = useState<Notice>(null);

  useEffect(() => {
    Promise.allSettled([api.health(), api.me()]).then(([healthResult, meResult]) => {
      if (healthResult.status === "fulfilled") setCapabilities(healthResult.value.capabilities);
      if (meResult.status === "fulfilled") setUser(meResult.value);
      setBooting(false);
    });
    const onHashChange = () => setProjectId(routeProjectId());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  useEffect(() => {
    if (!notice) return;
    const timeout = window.setTimeout(() => setNotice(null), 3600);
    return () => window.clearTimeout(timeout);
  }, [notice]);

  function openProject(id: string) {
    window.location.hash = `project=${id}`;
  }

  function closeProject() {
    window.location.hash = "";
  }

  async function logout() {
    await api.logout();
    closeProject();
    setUser(null);
  }

  if (booting) return <BootScreen />;

  return (
    <>
      {!user ? (
        <AuthScreen onAuthenticated={setUser} showNotice={setNotice} />
      ) : projectId ? (
        <ProjectWorkspace
          projectId={projectId}
          capabilities={capabilities}
          onBack={closeProject}
          showNotice={setNotice}
        />
      ) : (
        <LibraryScreen
          user={user}
          capabilities={capabilities}
          onOpen={openProject}
          onLogout={logout}
          showNotice={setNotice}
        />
      )}
      {notice && (
        <div className={`toast ${notice.tone}`} role="status">
          {notice.tone === "success" ? <CheckCircle2 size={18} /> : <ScanLine size={18} />}
          {notice.message}
        </div>
      )}
    </>
  );
}

function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`brand ${compact ? "compact" : ""}`}>
      <span className="brand-mark" aria-hidden="true">
        <Waves size={compact ? 17 : 21} />
      </span>
      <span>
        <strong>Nocturne</strong>
        {!compact && <small>夜谱</small>}
      </span>
    </div>
  );
}

function BootScreen() {
  return (
    <main className="boot-screen">
      <Brand />
      <LoaderCircle className="spin" size={22} />
    </main>
  );
}

function AuthScreen({
  onAuthenticated,
  showNotice
}: {
  onAuthenticated: (user: User) => void;
  showNotice: (notice: Notice) => void;
}) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    try {
      const result =
        mode === "register"
          ? await api.register({ email, display_name: displayName, password })
          : await api.login({ email, password });
      onAuthenticated(result);
      showNotice({ message: mode === "register" ? "曲库已经为你准备好" : "欢迎回来", tone: "success" });
    } catch (error) {
      showNotice({ message: error instanceof Error ? error.message : "登录失败", tone: "error" });
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="auth-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <section className="auth-story">
        <Brand />
        <div className="auth-copy">
          <p className="eyebrow"><span /> 为练习而生的私人曲库</p>
          <h1>让视频里的谱，<br />跟着音乐流动。</h1>
          <p className="lead">
            收好每一份谱，校准每一个小节。下次打开时，从上次停下的位置继续。
          </p>
        </div>
        <div className="signal-window" aria-hidden="true">
          <div className="signal-topline"><span /><span /><span /></div>
          <div className="notation-lines">
            {[0, 1, 2, 3, 4, 5].map((line) => <i key={line} />)}
            <b className="playhead" />
            <em className="note note-a">7</em>
            <em className="note note-b">9</em>
            <em className="note note-c">10</em>
            <em className="note note-d">7</em>
          </div>
          <div className="signal-caption"><Music2 size={14} /> 02:18 · 第 24 小节</div>
        </div>
        <p className="auth-footnote">Private by default · 你的项目默认只对自己可见</p>
      </section>

      <section className="auth-panel-wrap">
        <div className="auth-panel">
          <div className="segmented" aria-label="登录方式">
            <button type="button" className={mode === "login" ? "active" : ""} onClick={() => setMode("login")}>登录</button>
            <button type="button" className={mode === "register" ? "active" : ""} onClick={() => setMode("register")}>创建账号</button>
          </div>
          <div className="form-heading">
            <h2>{mode === "login" ? "继续你的练习" : "建立私人曲库"}</h2>
            <p>{mode === "login" ? "项目、谱面和同步点都在原处。" : "现在只需要邮箱，稍后可以完善资料。"}</p>
          </div>
          <form onSubmit={submit} className="auth-form">
            {mode === "register" && (
              <label>
                <span>昵称</span>
                <div className="input-wrap"><UserRound size={17} /><input autoComplete="name" value={displayName} onChange={(e) => setDisplayName(e.target.value)} placeholder="琴友怎么称呼你" required /></div>
              </label>
            )}
            <label>
              <span>邮箱</span>
              <div className="input-wrap"><Library size={17} /><input type="email" autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="name@example.com" required /></div>
            </label>
            <label>
              <span>密码</span>
              <div className="input-wrap"><LockKeyhole size={17} /><input type="password" minLength={mode === "register" ? 8 : 1} autoComplete={mode === "register" ? "new-password" : "current-password"} value={password} onChange={(e) => setPassword(e.target.value)} placeholder={mode === "register" ? "至少 8 位" : "输入密码"} required /></div>
            </label>
            <button className="primary-button full" type="submit" disabled={busy}>
              {busy ? <LoaderCircle size={18} className="spin" /> : mode === "login" ? "进入曲库" : "创建并进入"}
              {!busy && <ChevronRight size={18} />}
            </button>
          </form>
          <p className="privacy-note"><LockKeyhole size={13} /> 密码经强哈希保存；登录凭证仅存于 HttpOnly Cookie。</p>
        </div>
      </section>
    </main>
  );
}

function LibraryScreen({
  user,
  capabilities,
  onOpen,
  onLogout,
  showNotice
}: {
  user: User;
  capabilities: Capabilities;
  onOpen: (id: string) => void;
  onLogout: () => void;
  showNotice: (notice: Notice) => void;
}) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [source, setSource] = useState("");
  const [title, setTitle] = useState("");
  const [rights, setRights] = useState(false);

  useEffect(() => {
    api.projects()
      .then(setProjects)
      .catch((error) => showNotice({ message: error.message, tone: "error" }))
      .finally(() => setLoading(false));
  }, [showNotice]);

  async function createProject(event: FormEvent) {
    event.preventDefault();
    setCreating(true);
    try {
      let project = await api.createProject({ source_input: source, title, rights_confirmed: rights });
      let notice: Notice = { message: "项目已保存到私人曲库", tone: "success" };
      if (capabilities.yt_dlp) {
        try {
          project = await api.inspectProject(project.id);
          notice = { message: "视频信息与封面已自动保存", tone: "success" };
        } catch (error) {
          notice = {
            message: `项目已创建，但封面获取失败：${error instanceof Error ? error.message : "可进入项目后重试"}`,
            tone: "error"
          };
        }
      }
      setProjects((current) => [project, ...current]);
      showNotice(notice);
      onOpen(project.id);
    } catch (error) {
      showNotice({ message: error instanceof Error ? error.message : "创建失败", tone: "error" });
    } finally {
      setCreating(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <Brand compact />
        <nav><span className="nav-current"><Library size={15} /> 曲库</span></nav>
        <div className="account-pill"><span>{user.display_name.slice(0, 1).toUpperCase()}</span><div><strong>{user.display_name}</strong><small>{user.email}</small></div><button type="button" onClick={onLogout} aria-label="退出登录"><LogOut size={17} /></button></div>
      </header>

      <div className="library-content">
        <section className="library-intro">
          <div>
            <p className="eyebrow"><span /> PERSONAL LIBRARY</p>
            <h1>晚上好，{user.display_name}</h1>
            <p>从一条视频开始，把谱留在自己的曲库里。</p>
          </div>
          <CapabilityLine capabilities={capabilities} />
        </section>

        <section className="source-composer">
          <div className="composer-icon"><Link2 size={22} /></div>
          <form onSubmit={createProject}>
            <div className="composer-main">
              <label htmlFor="source-input">B 站链接、BV 或 AV 号</label>
              <input id="source-input" value={source} onChange={(e) => setSource(e.target.value)} placeholder="例如 BV1xx411c7mD" required />
            </div>
            <div className="composer-title">
              <label htmlFor="title-input">项目名，可稍后修改</label>
              <input id="title-input" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="不填则使用视频标题" />
            </div>
            <button className="primary-button" type="submit" disabled={creating}>
              {creating ? <LoaderCircle size={18} className="spin" /> : <Plus size={18} />}
              新建项目
            </button>
            <label className="rights-check">
              <input type="checkbox" checked={rights} onChange={(e) => setRights(e.target.checked)} />
              <span><Check size={12} /></span>
              我确认有权处理该视频；未勾选仍可建草稿，但不能下载。
            </label>
          </form>
        </section>

        <section className="project-section">
          <div className="section-heading"><div><h2>我的曲库</h2><p>{projects.length ? `${projects.length} 个私人项目` : "准备收下你的第一首曲子"}</p></div></div>
          {loading ? (
            <div className="library-loading"><LoaderCircle className="spin" size={20} /> 正在读取曲库</div>
          ) : projects.length ? (
            <div className="project-grid">
              {projects.map((project) => (
                <button className="project-card" key={project.id} onClick={() => onOpen(project.id)} type="button">
                  <div className="project-cover">
                    <ProjectCover project={project} />
                    <span className={`status-badge ${project.status}`}>{STATUS_LABELS[project.status] ?? project.status}</span>
                  </div>
                  <div className="project-card-body">
                    <div><h3>{project.title}</h3><p>{project.source_metadata?.uploader || project.source_id}</p></div>
                    <ArrowUpRight size={18} />
                  </div>
                  <div className="project-meta"><span><Clock3 size={13} /> {formatDate(project.updated_at)}</span><span>{project.sync_points.length} 个同步点</span></div>
                </button>
              ))}
            </div>
          ) : (
            <div className="empty-library"><div className="empty-glyph"><BookOpen size={30} /></div><h3>曲库还是空的</h3><p>在上方粘贴一个 B 站吉他谱视频，建立第一份练习项目。</p></div>
          )}
        </section>
      </div>
    </main>
  );
}

function CapabilityLine({ capabilities }: { capabilities: Capabilities }) {
  const items = [
    ["视频解析", capabilities.yt_dlp && capabilities.ffmpeg],
    ["图片转 PDF", true],
    ["六线 TAB 识别", capabilities.tab_ocr],
    ["音频节拍分析", capabilities.audio_analysis],
    ["五线谱识别", capabilities.audiveris]
  ] as const;
  return (
    <div className="capability-line">
      {items.map(([label, available]) => <span key={label} className={available ? "ready" : "waiting"}><i />{label}{available ? "就绪" : "待安装"}</span>)}
    </div>
  );
}

function ProjectWorkspace({
  projectId,
  capabilities,
  onBack,
  showNotice
}: {
  projectId: string;
  capabilities: Capabilities;
  onBack: () => void;
  showNotice: (notice: Notice) => void;
}) {
  const [project, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);
  const [action, setAction] = useState<string | null>(null);
  const [title, setTitle] = useState("");
  const [measure, setMeasure] = useState(1);
  const [currentTime, setCurrentTime] = useState(0);
  const [follow, setFollow] = useState(false);
  const [backingVolume, setBackingVolume] = useState(0.82);
  const [scoreVolume, setScoreVolume] = useState(0.72);
  const [audioSource, setAudioSource] = useState<"uploaded" | "video">("uploaded");
  const [recognition, setRecognition] = useState<RecognitionDiagnostics | null>(null);
  const [reviewMeasure, setReviewMeasure] = useState(1);
  const [viewport, setViewport] = useState<HTMLDivElement | null>(null);
  const [viewMode, setViewMode] = useState<"video" | "frames" | "images" | "score" | "review">("video");
  const audioRef = useRef<HTMLAudioElement>(null);
  const initializedViewRef = useRef(false);
  const previousStatusRef = useRef<string | null>(null);
  const viewportCallback = useCallback((node: HTMLDivElement | null) => setViewport(node), []);

  const refresh = async () => {
    const value = await api.project(projectId);
    setProject(value);
    setTitle(value.title);
    if (!initializedViewRef.current) {
      setViewMode(
        value.score_file_url
          ? "score"
          : value.score_images.length
          ? "images"
          : value.video_frames.length
          ? "frames"
          : "video"
      );
      initializedViewRef.current = true;
    } else if (previousStatusRef.current === "analyzing" && value.video_frames.length) {
      setViewMode("frames");
    }
    previousStatusRef.current = value.status;
    setMeasure(Math.max(1, ...value.sync_points.map((point) => point.measure_number + 1)));
    return value;
  };

  useEffect(() => {
    initializedViewRef.current = false;
    previousStatusRef.current = null;
    setRecognition(null);
    setLoading(true);
    refresh()
      .catch((error) => showNotice({ message: error.message, tone: "error" }))
      .finally(() => setLoading(false));
  }, [projectId]);

  useEffect(() => {
    if (!project || (!["downloading", "analyzing", "recognizing"].includes(project.status) && project.audio_analysis?.status !== "pending")) return;
    const timer = window.setInterval(() => refresh().catch(() => undefined), 2500);
    return () => window.clearInterval(timer);
  }, [project?.status, project?.audio_analysis?.status]);

  useEffect(() => {
    if (audioRef.current) audioRef.current.volume = backingVolume;
  }, [audioSource, backingVolume, project?.audio_url, project?.video_url]);

  useEffect(() => {
    setAudioSource(project?.audio_url ? "uploaded" : "video");
  }, [project?.audio_url]);

  async function run(key: string, operation: () => Promise<unknown>, success?: string) {
    setAction(key);
    try {
      await operation();
      const refreshed = await refresh();
      if (key === "analyze" && refreshed.status !== "analyzing" && refreshed.video_frames.length) {
        setViewMode("frames");
      }
      if (success) showNotice({ message: success, tone: "success" });
    } catch (error) {
      showNotice({ message: error instanceof Error ? error.message : "操作失败", tone: "error" });
    } finally {
      setAction(null);
    }
  }

  async function saveTitle() {
    const trimmed = title.trim();
    if (!project || !trimmed || trimmed === project.title) return;
    await run("rename", () => api.renameProject(project.id, trimmed), "名称已保存");
  }

  async function openReviewAt(timeSeconds?: number) {
    if (!project) return;
    setAction("load-review");
    try {
      const diagnostics = recognition ?? await api.recognition(project.id);
      setRecognition(diagnostics);
      let target = diagnostics.summary.start_measure ?? diagnostics.measures[0]?.number ?? 1;
      if (typeof timeSeconds === "number" && diagnostics.frames.length) {
        const frame = [...diagnostics.frames].sort(
          (left, right) => Math.abs(left.time_seconds - timeSeconds) - Math.abs(right.time_seconds - timeSeconds)
        )[0];
        if (frame.start_measure !== null) target = frame.start_measure + (frame.highlighted_index ?? 0);
      }
      setReviewMeasure(target);
      setViewMode("review");
    } catch (error) {
      showNotice({ message: error instanceof Error ? error.message : "无法打开识别校对", tone: "error" });
    } finally {
      setAction(null);
    }
  }

  async function saveReviewedMeasure(measureNumber: number, events: RecognitionEvent[]) {
    if (!project) return;
    setAction("review-save");
    try {
      const diagnostics = await api.updateRecognitionMeasure(project.id, measureNumber, events);
      setRecognition(diagnostics);
      await refresh();
      showNotice({ message: `第 ${measureNumber} 小节已保存并重新生成乐谱`, tone: "success" });
    } catch (error) {
      showNotice({ message: error instanceof Error ? error.message : "小节保存失败", tone: "error" });
    } finally {
      setAction(null);
    }
  }

  function currentScorePosition() {
    if (!viewport) return 0;
    const maxScroll = viewport.scrollHeight - viewport.clientHeight;
    return maxScroll > 0 ? viewport.scrollTop / maxScroll : 0;
  }

  async function addPoint() {
    if (!project || !audioRef.current) return;
    const point = await api.addSyncPoint(project.id, {
      measure_number: measure,
      time_seconds: audioRef.current.currentTime,
      score_position: currentScorePosition(),
      label: ""
    });
    setProject({
      ...project,
      sync_points: [...project.sync_points, point].sort((a, b) => a.time_seconds - b.time_seconds)
    });
    setMeasure((value) => value + 1);
    showNotice({ message: `第 ${point.measure_number} 小节同步点已保存`, tone: "success" });
  }

  function updateFollow(time: number) {
    setCurrentTime(time);
    if (!follow || !project || !viewport || project.sync_points.length < 2) return;
    const position = interpolateScorePosition(project.sync_points, time);
    if (position === null) return;
    const maxScroll = viewport.scrollHeight - viewport.clientHeight;
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    viewport.scrollTo({ top: position * maxScroll, behavior: reduceMotion ? "auto" : "smooth" });
  }

  if (loading || !project) {
    return <main className="boot-screen"><LoaderCircle className="spin" size={22} /><span>正在打开项目</span></main>;
  }

  const hasVisualScore = project.score_images.length > 0 || Boolean(project.score_file_url);
  const practiceAudioUrl = audioSource === "uploaded"
    ? project.audio_url || project.video_url
    : project.video_url || project.audio_url;
  const effectiveAudioSource = audioSource === "video" && project.video_url ? "video" : project.audio_url ? "uploaded" : "video";
  const analyzedSourceMatches = project.audio_analysis?.source === (effectiveAudioSource === "uploaded" ? "uploaded_audio" : "video_audio");
  const canAlign = Boolean(practiceAudioUrl && hasVisualScore);
  const canReview = project.recognition_summary?.engine === "tab_cv_tesseract";
  const stageTitle =
    viewMode === "video"
      ? "视频选段与谱面框选"
      : viewMode === "frames"
      ? "切片候选帧"
      : viewMode === "review"
      ? "原帧与识别谱校对"
      : viewMode === "score"
      ? "可播放乐谱"
      : "谱面校准";

  return (
    <main className="workspace-shell">
      <header className="workspace-topbar">
        <button type="button" className="back-button" onClick={onBack}><ArrowLeft size={17} /> 曲库</button>
        <div className="workspace-title">
          <input value={title} onChange={(event) => setTitle(event.target.value)} onBlur={saveTitle} onKeyDown={(event) => { if (event.key === "Enter") event.currentTarget.blur(); }} aria-label="项目名称" />
          <span><Save size={12} /> {action === "rename" ? "正在保存" : "自动保存到私人曲库"}</span>
        </div>
        <div className={`workspace-status ${project.status}`}><i />{STATUS_LABELS[project.status] ?? project.status}</div>
      </header>

      <div className="workflow-rail">
        {[
          { number: "01", label: "获取视频", complete: Boolean(project.video_url), pending: project.source_metadata ? "等待获取" : "先解析信息" },
          { number: "02", label: "切片分析", complete: project.video_frames.length > 0, pending: project.video_url ? "可以选段" : "等待视频" },
          { number: "03", label: "PDF 与乐谱", complete: Boolean(project.pdf_url || project.score_file_url), pending: "等待谱图" },
          { number: "04", label: "音画对齐", complete: project.sync_points.length >= 2, pending: "等待乐谱" }
        ].map(({ number, label, complete, pending }) => (
          <div className={`workflow-step ${complete ? "complete" : ""}`} key={number}>
            <span>{complete ? <Check size={13} /> : number}</span>
            <div><strong>{label}</strong><small>{complete ? "已准备" : pending}</small></div>
          </div>
        ))}
      </div>

      <div className="workspace-layout">
        <section className="score-stage">
          <div className="stage-toolbar">
            <div>
              <span className="stage-kicker">SCORE WORKSPACE</span>
              <h2>{stageTitle}</h2>
            </div>
            <div className="stage-actions">
              {(project.video_url || project.video_frames.length > 0 || project.score_images.length > 0 || project.score_file_url) && (
                <div className="small-segmented">
                  {project.video_url && <button type="button" className={viewMode === "video" ? "active" : ""} onClick={() => setViewMode("video")}>选段</button>}
                  {project.video_frames.length > 0 && <button type="button" className={viewMode === "frames" ? "active" : ""} onClick={() => setViewMode("frames")}>切片</button>}
                  {project.score_images.length > 0 && <button type="button" className={viewMode === "images" ? "active" : ""} onClick={() => setViewMode("images")}>谱图</button>}
                  {project.score_file_url && <button type="button" className={viewMode === "score" ? "active" : ""} onClick={() => setViewMode("score")}>可播放谱</button>}
                  {canReview && <button type="button" className={viewMode === "review" ? "active" : ""} onClick={() => void openReviewAt()}>校对</button>}
                </div>
              )}
              {project.pdf_url && <a className="quiet-button" href={`${project.pdf_url}?download=1`} download><FileText size={15} /> 导出 PDF</a>}
              <label className="quiet-button file-label"><ImagePlus size={15} /> 上传谱图<input type="file" accept="image/*" multiple onChange={(event) => { if (event.target.files?.length) run("images", async () => { await api.uploadImages(project.id, event.target.files!); setViewMode("images"); }, "PDF 已生成"); event.currentTarget.value = ""; }} /></label>
              <label className="quiet-button file-label"><FileMusic size={15} /> 导入乐谱<input type="file" accept=".gp,.gp3,.gp4,.gp5,.gpx,.musicxml,.xml,.mxl" onChange={(event) => { const file = event.target.files?.[0]; if (file) run("score", async () => { await api.uploadScore(project.id, file); setViewMode("score"); }, "结构化乐谱已载入"); event.currentTarget.value = ""; }} /></label>
            </div>
          </div>

          <div
            className={`score-viewport ${viewMode === "video" ? "video-workspace" : !hasVisualScore && !project.video_frames.length ? "empty" : ""}`}
            ref={viewportCallback}
            onPointerDown={() => follow && setFollow(false)}
          >
            {viewMode === "video" && project.video_url ? (
              <VideoSliceEditor
                videoUrl={project.video_url}
                durationHint={project.source_metadata?.duration || 0}
                analysis={project.video_analysis}
                busy={project.status === "analyzing" || action === "analyze"}
                onAnalyze={(request: VideoAnalysisRequest) => {
                  void run("analyze", () => api.analyzeVideo(project.id, request), "切片分析任务已开始");
                }}
              />
            ) : viewMode === "frames" && project.video_frames.length ? (
              <div className="video-frame-grid">
                {project.video_frames.map((frame, index) => (
                  <figure className="video-frame-card" key={frame.id}>
                    <img src={frame.url} alt={`候选切片第 ${index + 1} 张`} loading="lazy" draggable={false} />
                    <figcaption>
                      <span>#{String(index + 1).padStart(3, "0")}</span>
                      <time>{formatTime(frame.time_seconds)} <b>{frame.time_seconds.toFixed(3)}s</b></time>
                      <small>源帧 {frame.source_frame}</small>
                      {canReview && <button type="button" className="frame-review-button" onClick={() => void openReviewAt(frame.time_seconds)}><PencilLine size={12} /> 对照校对</button>}
                    </figcaption>
                  </figure>
                ))}
              </div>
            ) : viewMode === "review" && recognition ? (
              <ScoreReviewPanel
                project={project}
                diagnostics={recognition}
                measureNumber={reviewMeasure}
                busy={action === "review-save"}
                onMeasureChange={setReviewMeasure}
                onSave={saveReviewedMeasure}
              />
            ) : viewMode === "score" && project.score_file_url ? (
              <Suspense fallback={<div className="score-loading"><LoaderCircle size={20} className="spin" /> 正在载入乐谱引擎</div>}>
                <AlphaTabPlayer scoreUrl={`${project.score_file_url}?v=${encodeURIComponent(project.updated_at)}`} scrollElement={viewport} masterVolume={scoreVolume} fileBaseName={project.title} />
              </Suspense>
            ) : viewMode === "images" && project.score_images.length ? (
              <div className="score-pages">
                {project.score_images.map((image, index) => (
                  <figure className="score-page" key={image.id}>
                    <span>{String(index + 1).padStart(2, "0")}</span>
                    <img src={image.url} alt={`谱面第 ${index + 1} 页`} draggable={false} />
                  </figure>
                ))}
              </div>
            ) : project.video_url ? (
              <div className="video-waiting"><Scissors size={28} /><h3>视频已经准备好</h3><p>切换到“选段”，设置开始与结束时间，再在画面上框选谱面区域。</p><button className="primary-button" type="button" onClick={() => setViewMode("video")}>开始选段</button></div>
            ) : (
              <label className="score-dropzone">
                <span className="drop-icon"><Upload size={24} /></span>
                <h3>先解析并获取视频</h3>
                <p>视频准备完成后，这里会出现时间范围、画面框选和抽帧间隔工具；也可以直接上传谱图。</p>
                <strong>{action === "images" ? "正在生成 PDF…" : "选择谱图"}</strong>
                <input type="file" accept="image/*" multiple onChange={(event) => { if (event.target.files?.length) run("images", async () => { await api.uploadImages(project.id, event.target.files!); setViewMode("images"); }, "PDF 已生成"); event.currentTarget.value = ""; }} />
              </label>
            )}
          </div>
          {((action && action !== "rename" && action !== "rights") || ["downloading", "analyzing", "recognizing"].includes(project.status) || project.audio_analysis?.status === "pending") && <div className="stage-progress"><LoaderCircle size={15} className="spin" /> {project.status_message || "正在处理，请不要关闭页面"}</div>}
        </section>

        <aside className="inspector">
          <section className="inspector-section source-section">
            <div className="inspector-heading"><span><Link2 size={16} /></span><div><h3>视频来源</h3><p>{project.source_id}</p></div></div>
            {project.source_metadata ? (
              <div className="source-preview">
                {project.cover_url && <img src={project.cover_url} alt="" referrerPolicy="no-referrer" />}
                <div><strong>{project.source_metadata.title}</strong><span>{project.source_metadata.uploader || "Bilibili"} · {formatTime(project.source_metadata.duration)}</span></div>
              </div>
            ) : (
              <p className="inspector-copy">先读取标题、作者和时长；这一步不会下载视频。</p>
            )}
            {project.source_metadata && !project.video_url && <p className="inspector-copy">信息已经解析。确认处理权限并获取视频后，主工作台会显示选段与框选工具。</p>}
            {project.video_url && <p className="inspector-copy">视频已保存在私人项目中，可以回到主工作台选择时间和谱面范围。</p>}
            <div className="button-row">
              <button className="secondary-button" type="button" disabled={action === "inspect" || !capabilities.yt_dlp} onClick={() => run("inspect", () => api.inspectProject(project.id), "视频信息已解析；下一步请获取视频")}><RefreshCw size={15} className={action === "inspect" ? "spin" : ""} /> 解析信息</button>
              <button className="secondary-button" type="button" disabled={action === "download" || ["downloading", "analyzing"].includes(project.status) || !capabilities.yt_dlp || !capabilities.ffmpeg || !project.rights_confirmed} onClick={() => run("download", () => api.downloadProject(project.id), "下载任务已经开始")}><Download size={15} /> {project.video_url ? "重新获取" : "获取视频"}</button>
              <a className="icon-button" href={project.source_url} target="_blank" rel="noreferrer" aria-label="打开原视频"><ArrowUpRight size={16} /></a>
            </div>
            <label className={`project-rights-toggle ${project.rights_confirmed ? "confirmed" : ""}`}>
              <input
                type="checkbox"
                checked={project.rights_confirmed}
                disabled={action === "rights" || project.status === "downloading"}
                onChange={(event) => void run("rights", () => api.updateProjectRights(project.id, event.target.checked), event.target.checked ? "已启用视频处理" : "已撤销处理确认")}
              />
              <span><Check size={12} /></span>
              <small>{project.rights_confirmed ? "已确认有权处理这个视频" : "确认有权处理后才能获取视频"}</small>
            </label>
            {!project.rights_confirmed && <p className="inline-warning"><LockKeyhole size={13} /> 你创建草稿时没有勾选权限确认，现在可以在这里补充确认。</p>}
          </section>

          <section className="inspector-section">
            <div className="inspector-heading"><span><WandSparkles size={16} /></span><div><h3>乐谱识别</h3><p>{project.video_frames.length ? (capabilities.tab_ocr ? "六线 TAB 专用引擎已就绪" : "TAB OCR 引擎尚未安装") : (capabilities.audiveris ? "Audiveris 已就绪" : "五线谱引擎尚未安装")}</p></div></div>
            <p className="inspector-copy">{project.video_frames.length ? "从原始切片识别弦号、品位和八分音符网格，按小节号去重合成完整 PDF；校对器可用数字键改品位，并添加连音、滑音、击勾弦、推弦等技巧。" : "PDF 路线用于清晰印刷五线谱，识别后仍需逐小节校对。"}</p>
            {project.recognition_summary && <p className="inspector-copy">上次结果：{project.recognition_summary.engine_label}{project.recognition_summary.measure_count ? ` · ${project.recognition_summary.measure_count} 小节` : ""}{typeof project.recognition_summary.confidence === "number" ? ` · 数字置信度 ${Math.round(project.recognition_summary.confidence * 100)}%` : ""}</p>}
            <button className="secondary-button full" type="button" disabled={action === "recognize" || (project.video_frames.length ? !capabilities.tab_ocr : (!project.pdf_url || !capabilities.audiveris))} onClick={() => run("recognize", () => api.recognizeProject(project.id), "识别任务已经开始")}>
              {action === "recognize" ? <LoaderCircle size={15} className="spin" /> : <ScanLine size={15} />}
              {project.video_frames.length ? (capabilities.tab_ocr ? "识别视频六线 TAB" : "安装 TAB OCR 后可用") : (capabilities.audiveris ? "识别五线谱 PDF" : "安装 Audiveris 后可用")}
            </button>
          </section>

          <section className="inspector-section audio-section">
            <div className="inspector-heading"><span><Headphones size={16} /></span><div><h3>音频分析与混音</h3><p>{effectiveAudioSource === "uploaded" ? project.audio_name : (project.video_url ? "当前使用视频原声" : "上传音乐或伴奏")}</p></div></div>
            {project.audio_url && project.video_url && <div className="audio-source-switch small-segmented"><button type="button" className={effectiveAudioSource === "uploaded" ? "active" : ""} onClick={() => setAudioSource("uploaded")}>上传音频</button><button type="button" className={effectiveAudioSource === "video" ? "active" : ""} onClick={() => setAudioSource("video")}>视频原声</button></div>}
            {practiceAudioUrl ? (
              <audio ref={audioRef} src={practiceAudioUrl} controls preload="metadata" onTimeUpdate={(event) => updateFollow(event.currentTarget.currentTime)} onSeeked={(event) => updateFollow(event.currentTarget.currentTime)} />
            ) : (
              <label className="audio-upload"><AudioLines size={19} /><span><strong>选择音频</strong><small>MP3、WAV、M4A、OGG 等</small></span><input type="file" accept="audio/*,.flac" onChange={(event) => { const file = event.target.files?.[0]; if (file) run("audio", async () => { await api.uploadAudio(project.id, file); setAudioSource("uploaded"); }, "练习音频已保存"); event.currentTarget.value = ""; }} /></label>
            )}
            {practiceAudioUrl && <label className="replace-link"><Upload size={13} /> {project.audio_url ? "更换上传音频" : "上传 MP3 替代视频原声"}<input type="file" accept="audio/*,.flac" onChange={(event) => { const file = event.target.files?.[0]; if (file) run("audio", async () => { await api.uploadAudio(project.id, file); setAudioSource("uploaded"); }, "练习音频已更新"); event.currentTarget.value = ""; }} /></label>}

            <div className="audio-mixer" aria-label="音量混合器">
              <div className="mixer-title"><Volume2 size={14} /><span>对比音量</span><small>拖动时立即生效</small></div>
              <label><span>{effectiveAudioSource === "uploaded" ? "上传音频" : "视频原声"}</span><input type="range" min="0" max="1" step="0.01" value={backingVolume} disabled={!practiceAudioUrl} onChange={(event) => setBackingVolume(Number(event.target.value))} aria-label="伴奏音量" /><output>{Math.round(backingVolume * 100)}%</output></label>
              <label><span>识别谱音色</span><input type="range" min="0" max="1" step="0.01" value={scoreVolume} disabled={!project.score_file_url} onChange={(event) => setScoreVolume(Number(event.target.value))} aria-label="识别谱音量" /><output>{Math.round(scoreVolume * 100)}%</output></label>
            </div>

            <div className="audio-analysis-actions">
              <button className="secondary-button full" type="button" disabled={!capabilities.audio_analysis || !practiceAudioUrl || project.audio_analysis?.status === "pending" || action === "audio-analysis"} onClick={() => run("audio-analysis", () => api.analyzeAudio(project.id, effectiveAudioSource), "音频分析任务已开始")}>
                {project.audio_analysis?.status === "pending" ? <LoaderCircle size={15} className="spin" /> : <Waves size={15} />}
                {effectiveAudioSource === "uploaded" ? "分析上传音频" : "分析视频原声"}
              </button>
            </div>

            {project.audio_analysis?.status === "complete" && (
              <div className="audio-analysis-result">
                <div className="analysis-stats">
                  <span><strong>{project.audio_analysis.tempo_bpm?.toFixed(1)}</strong><small>BPM</small></span>
                  <span><strong>{project.audio_analysis.sections?.length ?? 0}</strong><small>段落候选</small></span>
                  <span><strong>{project.audio_analysis.alignment_suggestions?.length ?? 0}</strong><small>对齐建议</small></span>
                </div>
                <div className="section-timeline">
                  {(project.audio_analysis.sections ?? []).map((section, index) => (
                    <button type="button" key={`${section.label}-${index}`} onClick={() => { if (audioRef.current) { audioRef.current.currentTime = section.start_seconds; updateFollow(section.start_seconds); } }}>
                      <strong>{section.label}</strong><span>{formatTime(section.start_seconds)}–{formatTime(section.end_seconds)}</span>
                    </button>
                  ))}
                </div>
                <p>来自 {project.audio_analysis.source_label}。A/B/C 是曲式候选，可点击跳到该段试听，不会冒充确定的主歌或副歌。</p>
                {(project.audio_analysis.alignment_suggestions?.length ?? 0) >= 2 && <button className="secondary-button full apply-alignment" type="button" disabled={action === "apply-alignment" || !analyzedSourceMatches} onClick={() => run("apply-alignment", () => api.applyAudioAlignment(project.id), "自动对齐点已加入；已有手动点未被覆盖")}><ListMusic size={15} /> {analyzedSourceMatches ? "应用自动对齐建议" : `请切换到${project.audio_analysis.source === "uploaded_audio" ? "上传音频" : "视频原声"}`}</button>}
              </div>
            )}
            {project.audio_analysis?.status === "failed" && <p className="inline-warning">音频分析失败：{project.audio_analysis.error}</p>}
          </section>

          <section className="inspector-section sync-section">
            <div className="inspector-heading"><span><AudioLines size={16} /></span><div><h3>手动对齐</h3><p>{project.sync_points.length} 个同步点</p></div></div>
            <div className="sync-readout"><span>{formatTime(currentTime)}</span><i /><span>{Math.round(currentScorePosition() * 100)}% 谱面</span></div>
            <div className="sync-recorder">
              <label><span>小节</span><input type="number" min="1" value={measure} onChange={(event) => setMeasure(Number(event.target.value))} /></label>
              <button type="button" className="primary-button" disabled={!canAlign || action === "sync"} onClick={() => run("sync", addPoint)}><Plus size={16} /> 记录当前位置</button>
            </div>
            <p className="sync-help">播放到目标时间，再把谱滚到对应小节并记录。至少两个点后即可跟随。</p>
            <label className={`follow-toggle ${project.sync_points.length < 2 ? "disabled" : ""}`}>
              <span><strong>播放时跟随谱面</strong><small>在同步点之间平滑插值</small></span>
              <input type="checkbox" checked={follow} disabled={project.sync_points.length < 2} onChange={(event) => setFollow(event.target.checked)} />
              <i />
            </label>
            <div className="sync-list">
              {project.sync_points.map((point) => (
                <div className="sync-point" key={point.id}>
                  <span>#{point.measure_number}</span><time>{formatTime(point.time_seconds)}</time><em>{Math.round(point.score_position * 100)}%</em>
                  <button type="button" onClick={() => run("delete-point", async () => { await api.deleteSyncPoint(project.id, point.id); setProject({ ...project, sync_points: project.sync_points.filter((item) => item.id !== point.id) }); })} aria-label="删除同步点"><Trash2 size={13} /></button>
                </div>
              ))}
            </div>
          </section>

          <p className="inspector-status"><i className={project.status === "failed" ? "error" : ""} /> {project.status_message}</p>
        </aside>
      </div>
    </main>
  );
}

export default App;
