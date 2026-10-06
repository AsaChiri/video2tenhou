export type Point = [number, number];
export type Rectangle = [number, number, number, number];

/** A hand of the analysis as saved in hands.json. */
export interface HandEntry {
  hand: number;
  game: number;
  kyoku: number;
  honba: number;
  t_start: number;
  t_end: number;
  corner_wind: Record<string, string>;
  nicks?: Record<string, string>;
}
/** One row of the review's hand list. */
export interface HandRow {
  hand: number;
  game: number;
  kyoku: number;
  honba: number;
  t_start: number;
  t_end: number;
  status: string | null;
  pending: boolean;
  turns: number | null;
  score: boolean | null;
}

export interface ReviewChoice {
  id?: string;
  kind?: string;
  field?: string;
  seat?: string;
  j?: number;
  t?: number;
  value?: string | string[];
  lost?: boolean;
  margin?: number | null;
  alternative_gap?: number | null;
  runner_up?: string | null;
}
export interface ReviewItem extends ReviewChoice {
  id: string;
  kind: string;
  hand: number;
  game?: number;
  kyoku?: number;
  honba?: number;
  stage?: string;
  i?: number;
  tile?: string;
  tiles?: string[];
  seats?: string[];
  choices?: ReviewChoice[];
  text?: string;
  type?: string;
  source?: string | null;
  guess?: boolean | string[];
  culprit?: string | null;
  fact_seat?: string | null;
  fact_kind?: string | null;
  han?: number;
  fu?: number;
  site?: number[];
  same_payment?: boolean;
  win_tile?: string;
  candidates?: string[];
  over?: {
    tile: string;
    count: number;
    limit?: number;
    sources: EvidenceSource[];
  }[];
  violations?: Violation[];
}
/** One rule of play an exported log breaks; `seat` is the hand's wind. */
export interface Violation {
  kind: string;
  seat: string | null;
  tile: string | null;
  t: number | null;
  text: string;
}
export interface EvidenceSource {
  kind: string;
  seat: string;
  t: number;
  tile: string;
  corner: string;
  j?: number;
  field?: string;
  index?: number;
  type?: string;
  tiles?: string[];
  source?: string;
  pos?: Point;
  t_pic?: number;
  box?: Rectangle;
  call?: { corner: string; t: number };
}

/** A saved review answer (labels/<video>/facts.jsonl). */
export interface Fact {
  kind: string;
  hand: number;
  ts?: number;
  author?: string;
  source?: string;
  seat?: string;
  corner?: string;
  t?: number;
  t_discard?: number;
  j?: number;
  field?: string;
  tile?: string;
  tiles?: string[];
  seats?: string[];
  type?: string;
  called_pos?: number | null;
  item?: string;
  text?: string;
  han?: number;
  fu?: number;
  site?: number[];
}
export type FactBody = Omit<Fact, "hand" | "ts" | "author"> & { hand?: number };
export type Facts = Partial<Record<number, Fact[]>>;
/** One question of the queue: an item, or one choice of a grouped item. */
export interface Decision {
  key: string;
  item: ReviewItem;
  choice: ReviewChoice | null;
}
export interface IgnoredAnswer {
  ts: number | null;
  kind: string;
  reason: string;
}

/** The workspace's single processing job. */
export interface WorkspaceJob {
  kind: "prepare" | "analyze" | "rebuild" | "fit" | "check";
  project: string;
  stage: string;
  hands: number[] | null;
  running: boolean;
  started: number;
  finished: number | null;
  error: string | null;
  /** Total appended lines for this job; the fetched log retains only its tail. */
  log_lines: number;
}
export interface JobStatus {
  job: WorkspaceJob | null;
  revision: string | null;
}
/** A project's latest preparation or analysis. */
export interface ProjectJob {
  action?: string;
  running: boolean;
  stage?: string;
  started?: number;
  finished?: number | null;
  error?: string | null;
}
export interface Project {
  id: string;
  name: string;
  display_name: string;
  kind: string;
  source: string;
  games: number[];
  layout: string;
  start: number;
  end: number | null;
  created?: number;
  status: string;
  job: ProjectJob;
  checking: boolean;
  artifacts: string[];
  stale_exports: boolean;
  results_revision: [string, number, number][];
  can_calibrate: boolean;
  has_fit: boolean;
  open_items: number | null;
}
export interface Turn {
  i: number;
  j: number;
  seat: string;
  t: number;
  t_prev?: number;
  discard: string;
  draw?: string;
  call?: { type: string; tiles: string[]; seat: string };
  own_call?: { type: string; tiles: string[]; seat: string };
  hand_before?: string[];
  hand_after?: string[];
  riichi?: boolean;
  tsumogiri?: boolean;
  discard_box?: Rectangle;
}
export interface Decode {
  confidence?: (ReviewChoice & { turn?: number })[];
  t_last?: number;
  play_window?: Point;
  dealer?: string;
  dora?: string[];
  ura?: string[];
  haipai?: Record<string, string[]>;
  turns?: Turn[];
  calls?: {
    seat: string;
    type: string;
    tiles: string[];
    t_first: number;
    source?: string;
  }[];
  indicators?: {
    tile: string;
    region: string;
    t_first: number;
    box?: Rectangle;
  }[];
  riichi?: string[];
  result?: { outcome?: string; winner?: string; loser?: string };
  notes?: string[];
}
export interface HandData {
  entry: HandEntry;
  decode: Decode | null;
  ignored: IgnoredAnswer[];
}
export interface LabelBox {
  xyxy: Rectangle;
  tile: string;
  sideways: boolean;
  role: string;
}
export interface Reading {
  boxes: LabelBox[];
  size: Point;
}
export interface CameraData {
  read_before?: { tiles?: { tile: string }[] }[];
  read_after?: { tiles?: { tile: string }[] }[];
  turn?: Turn;
}
export interface Results {
  games: {
    index: number;
    names: string[];
    viewer_url: string;
    download: string;
    hands: {
      index: number;
      round: string;
      honba: number;
      editor_url: string;
    }[];
  }[];
  pending_games: number[];
}
export interface Workspace {
  projects: Project[];
  setup: { ready: boolean; missing: string[] };
}
export interface CalibrationRegion {
  kind: string;
  movable: boolean;
  rect: Rectangle;
  quad: Point[];
  roll?: number | null;
}
export interface Overhead {
  center: Point;
  angle: number;
  scale: number;
}
export interface Calibration {
  layout?: string;
  frame: Point;
  regions: Record<string, CalibrationRegion>;
  fit?: { overhead?: Overhead } | null;
  overhead?: Overhead;
  checks?: Record<string, { level: string }>;
}
export interface Drag {
  name: string;
  mode: string;
  i?: number;
  x: number;
  y: number;
  rect: Rectangle;
}
