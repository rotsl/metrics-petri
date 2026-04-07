export type AnalysisEngine = 'local' | 'gemini';

export interface InputImage {
  filename: string;
  day: number;
  imageUrl: string;
  selected: boolean;
}

export interface AnalysisResult {
  id: string;
  filename: string;
  day: number;
  imageUrl: string;
  rawAnalysis?: {
    dish_center?: {x: number; y: number};
    dish_radius?: number;
    colony_polygon?: Array<{x: number; y: number}>;
    cracks?: Array<Array<{x: number; y: number}>>;
    internal_band_description?: string;
    radial_profile?: {
      radiusFraction?: number[];
      radiusMm?: number[];
      meanIntensity?: number[];
      ringSpacingMm?: number;
      centerToEdgeDelta?: number;
      densityIndex?: number;
    };
    crack_analysis?: {
      analysis_band_mm?: number;
      analysis_threshold?: number;
      crack_area_px?: number;
      total_length_px?: number;
      num_segments?: number;
      mean_segment_length_px?: number;
    };
    morphology_estimates?: {
      area_mm2?: number;
      perimeter_mm?: number;
      diameter_mm?: number;
    };
    segmentation_diagnostics?: {
      petri_dish_diameter_mm?: number;
      initial_mask_area_px?: number;
      classical_mask_area_px?: number;
      unet_mask_area_px?: number;
      hybrid_pre_sam_area_px?: number;
      refined_mask_area_px?: number;
      refinement_area_ratio?: number;
      refinement_iou_with_model_prior?: number;
      refinement_centroid_shift_px?: number;
      classical_unet_iou?: number;
      classical_unet_size_ratio?: number;
      hybrid_strategy?: string;
      hybrid_strategy_reason?: string;
      gemma_prior_iou_with_hybrid?: number;
      gemma_prior_area_px?: number;
      base_hybrid_area_px?: number;
      gemma_prior_area_ratio_vs_hybrid?: number;
      gemma_blend_decision?: string;
      sam_decision?: string;
      sam_error?: string;
      sam_area_px?: number;
      sam_final_area_px?: number;
      sam_iou_with_base?: number;
      sam_iou_with_classical?: number;
      sam_area_ratio_vs_base?: number;
    };
  };
  pixelToMm: number;
  morphology: {
    areaMm2: number;
    equivalentRadiusMm: number;
    diameterMm: number;
    perimeterMm: number;
    circularity: number;
    eccentricity: number;
    edgeRoughness: number;
  };
  texture: {
    contrast: number;
    correlation: number;
    energy: number;
    homogeneity: number;
    entropy: number;
    centerToEdgeDelta: number;
    densityIndex: number;
    radialZonation: {
      core: number;
      middle: number;
      outer: number;
    };
  };
  cracks: {
    count: number;
    totalLengthMm: number;
    coveragePct: number;
    proportionalCoveragePct: number;
    internalBandSummary: string;
  };
  kinematics: {
    radialVelocity: number;
    areaGrowthRate: number;
    relativeGrowthRate: number;
    radialAcceleration: number;
  };
  qcStatus: 'pass' | 'fail' | 'warning';
  qcNotes: string;
}

export interface AnalysisRun {
  id?: string;
  engine: AnalysisEngine;
  engineModel: string;
  createdAt: string;
  outputDir: string;
  analysisJson: string;
  analysisCsv: string;
  experimentName?: string;
  tags?: string[];
}

export interface OutputRunPayload {
  run: AnalysisRun;
  results: AnalysisResult[];
}

export interface ReportBundle {
  runId: string;
  template: string;
  generatedAt: string;
  markdownPath: string;
  pdfPath: string;
  assetsDir: string;
  markdownContent: string;
  graphCount: number;
  experimentName?: string;
  tags?: string[];
}

export interface AnalysisJobProgress {
  current: number;
  total: number;
  stage: string;
}

export interface AnalysisJob {
  id: string;
  engine: AnalysisEngine;
  filenames: string[];
  experimentName?: string;
  tags?: string[];
  status: 'queued' | 'running' | 'paused' | 'completed' | 'failed' | 'stopped';
  createdAt: string;
  updatedAt: string;
  progress: AnalysisJobProgress;
  logs: string[];
  result?: OutputRunPayload;
  error?: string;
}

export interface GrowthData {
  day: number;
  area: number;
  radius: number;
  diameter: number;
  perimeter: number;
  circularity: number;
  eccentricity: number;
  edgeRoughness: number;
  contrast: number;
  entropy: number;
  centerEdgeDelta: number;
  densityIndex: number;
  ringSpacing: number;
  crackCoverage: number;
  proportionalCrackCoverage: number;
  crackCount: number;
  velocity: number;
  areaRate: number;
  relativeGrowthRate: number;
  acceleration: number;
  core: number;
  middle: number;
  outer: number;
  outerCoreDelta: number;
  middleCoreDelta: number;
  textureSpread: number;
}
