export type Point = [number, number];
export type Rectangle = [number, number, number, number];

export interface HandEntry {
  hand: number;
  kyoku: number;
  honba: number;
  corner_wind?: Record<string, string>;
  nicks?: Record<string, string>;
  decoded_at?: number;
  pending_rebuild?: boolean;
  facts_newer?: number;
  t_start?: number;
  t_end?: number;
  status?: string;
  game: number;
  turns?: number;
  score?: boolean;
  t_last?: number;
  play_window?: Point;
}

export interface ReviewItem {
  kind: string;
  stage?: string;
  hand: number;
  idx?: number;
  field?: string;
  seat?: string;
  j?: number;
  i?: number;
  t?: number;
  tile?: string;
  tiles?: string[];
  seats?: string[];
  choices?: ReviewChoice[];
  lost?: boolean;
  margin?: number | null;
  alternative_gap?: number | null;
  runner_up?: string | null;
  text?: string;
  type?: string;
  source?: string | null;
  ts?: number;
  author?: string;
  t_discard?: number;
  called_pos?: number | null;
  value?: string | string[];
  guess?: boolean;
  culprit?: string;
  han?: number;
  fu?: number;
  site?: number[];
  same_payment?: boolean;
  outcome?: string;
  conflict?: boolean;
  box?: Rectangle;
  over?: {
    tile: string;
    count: number;
    limit?: number;
    sources: EvidenceSource[];
  }[];
}
export type ReviewChoice = Omit<ReviewItem, "hand" | "kind"> & {
  hand?: number;
  kind?: string;
};
export interface EvidenceSource {
  kind: string;
  seat: string;
  t: number;
  tile: string;
  corner: string;
  type?: string;
  tiles?: string[];
  source?: string;
  pos?: Point;
  t_pic?: number;
  box?: Rectangle;
  call?: { corner: string; t: number };
}

export type Fact = ReviewItem;
export type FactBody = Omit<Fact, "hand"> & { hand?: number };
export type Facts = Partial<Record<number, Fact[]>>;
export interface Decision {
  item: ReviewItem;
  choice: number | null;
  decision: ReviewChoice;
}
export interface Job {
  running: boolean;
  hands?: number[];
  pending?: number[];
  error?: string;
  hands_done?: number;
  hands_total?: number;
  log?: string[];
  kind?: string;
  started?: number;
  finished?: number;
  stage?: string;
}
export interface Project {
  id: string;
  name: string;
  display_name?: string;
  created?: number;
  review_running?: boolean;
  artifacts: string[];
  job: Job;
  has_fit: boolean;
  can_calibrate?: boolean;
  open_items: number;
  source: string;
  games: number[];
  start: string;
  end: string;
  layout: string;
  stale_exports?: boolean;
  results_revision?: string;
  pending_rebuilds?: number[];
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
  margin?: number;
  discard_box?: Rectangle;
}
export interface Decode {
  confidence?: (ReviewChoice & { turn?: number })[];
  t_last?: number;
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
  problems?: string[];
}
export interface HandData {
  entry: HandEntry;
  decode?: Decode;
  notes?: string[];
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
  roll?: number;
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
  fit?: { overhead?: Overhead };
  overhead?: Overhead;
  checks?: Record<string, { level: string; held: number; cut: number }>;
}
export interface Drag {
  name: string;
  mode: string;
  i?: number;
  x: number;
  y: number;
  rect: Rectangle;
}
