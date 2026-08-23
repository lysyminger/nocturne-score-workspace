export type User = {
  id: number;
  email: string;
  display_name: string;
  created_at?: string;
};

export type Capabilities = {
  ffmpeg: boolean;
  yt_dlp: boolean;
  audiveris: boolean;
  tab_ocr: boolean;
  audio_analysis: boolean;
};

export type RecognitionSummary = {
  engine: string;
  engine_label: string;
  measure_count?: number;
  start_measure?: number;
  end_measure?: number;
  estimated_tempo_bpm?: number;
  confidence?: number;
  low_confidence_glyphs?: number;
  warnings?: string[];
};

export type SyncPoint = {
  id: number;
  project_id: string;
  measure_number: number;
  time_seconds: number;
  score_position: number;
  label: string;
  created_at: string;
};

export type AudioSection = {
  label: string;
  start_seconds: number;
  end_seconds: number;
  confidence: number;
};

export type AlignmentSuggestion = {
  measure_number: number;
  time_seconds: number;
  score_position: number;
  label: string;
  confidence: number;
  basis: "video_highlight" | "audio_beat_grid";
};

export type AudioAnalysis = {
  status: "pending" | "complete" | "failed";
  engine?: string;
  source: "uploaded_audio" | "video_audio";
  source_label: string;
  duration_seconds?: number;
  tempo_bpm?: number;
  tempo_confidence?: number;
  beat_count?: number;
  onset_count?: number;
  beat_times?: number[];
  onset_times?: number[];
  sections?: AudioSection[];
  alignment_suggestions?: AlignmentSuggestion[];
  warnings?: string[];
  started_at?: string;
  completed_at?: string;
  error?: string;
};

export type TabTechnique =
  | "legato"
  | "slide"
  | "hammer_on"
  | "pull_off"
  | "bend"
  | "vibrato"
  | "harmonic"
  | "palm_mute"
  | "let_ring"
  | "dead_note";

export type RecognitionNote = {
  string: number;
  fret: number;
  technique?: TabTechnique;
};

export type RecognitionEvent = {
  onset_eighths: number;
  duration_eighths: number;
  notes: RecognitionNote[];
};

export type RecognitionMeasure = {
  number: number;
  quality: number;
  source_time: number;
  events: RecognitionEvent[];
};

export type RecognitionFrame = {
  name: string;
  time_seconds: number;
  start_measure: number | null;
  start_measure_confidence: number;
  highlighted_index: number | null;
  raw_measure_labels: Array<string | null>;
};

export type RecognitionDiagnostics = {
  summary: RecognitionSummary;
  sync_suggestions: Array<{ measure_number: number; time_seconds: number }>;
  frames: RecognitionFrame[];
  measures: RecognitionMeasure[];
};

export type ScoreImage = {
  id: string;
  original_name: string;
  media_type: string;
  sort_order: number;
  created_at: string;
  url: string;
};

export type CropRegion = {
  crop_x: number;
  crop_y: number;
  crop_width: number;
  crop_height: number;
};

export type VideoAnalysisRequest = CropRegion & {
  start_seconds: number;
  end_seconds: number;
  frame_interval: number;
};

export type VideoAnalysis = VideoAnalysisRequest & {
  analysis_id: string;
  status: "pending" | "complete" | "failed";
  source_fps: number;
  source_width: number;
  source_height: number;
  estimated_frames?: number;
  frame_count?: number;
  preview_pdf_status?: "complete" | "failed";
  preview_pdf_error?: string | null;
  created_at: string;
  completed_at?: string;
  error?: string;
};

export type VideoFrame = ScoreImage & {
  time_seconds: number;
  source_frame: number;
};

export type SourceMetadata = {
  id: string;
  title: string;
  uploader: string;
  duration: number;
  thumbnail: string;
  webpage_url: string;
  extractor: string;
};

export type Project = {
  id: string;
  title: string;
  source_input: string;
  source_kind: "bv" | "av" | "manual_tab";
  source_id: string;
  source_url: string;
  rights_confirmed: boolean;
  status: string;
  status_message: string;
  source_metadata: SourceMetadata | null;
  score_file_name: string | null;
  audio_name: string | null;
  created_at: string;
  updated_at: string;
  sync_points: SyncPoint[];
  score_images: ScoreImage[];
  video_frames: VideoFrame[];
  video_analysis: VideoAnalysis | null;
  recognition_summary: RecognitionSummary | null;
  audio_analysis: AudioAnalysis | null;
  cover_url: string | null;
  pdf_url: string | null;
  audio_url: string | null;
  score_file_url: string | null;
  video_url: string | null;
};
