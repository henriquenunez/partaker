# Complete Partaker 2 Workflow - From Microscopy to Environmental Correlation

## Paper Objective
**Analyze how bacterial cell morphology and behavior correlate with microfluidic environmental conditions (velocity, pressure, shear stress)**

---

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          DATA ACQUISITION PHASE                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝

    ┌─────────────────────┐              ┌─────────────────────┐
    │  ND2 Microscopy     │              │  COMSOL FEM         │
    │  Time-Lapse Images  │              │  Simulation Results │
    │                     │              │                     │
    │  ✅ EXISTS          │              │  ⚠️  TO INTEGRATE   │
    │  - Phase contrast   │              │  - Velocity field   │
    │  - mCherry (opt)    │              │  - Pressure field   │
    │  - YFP (opt)        │              │  - Shear stress     │
    └──────────┬──────────┘              └──────────┬──────────┘
               │                                    │
               ▼                                    ▼

╔═══════════════════════════════════════════════════════════════════════════════╗
║                       DATA LOADING & PREPROCESSING                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝

    ┌─────────────────────┐              ┌─────────────────────┐
    │  ImageData Singleton│              │  ComsiolDataLoader  │
    │  ✅ EXISTS          │              │  ❌ TO BUILD        │
    │                     │              │                     │
    │  • Dask lazy load   │              │  • Parse FEM output │
    │  • Multi-file concat│              │  • Extract fields   │
    │  • TPCYX format     │              │  • Grid coordinates │
    └──────────┬──────────┘              └──────────┬──────────┘
               │                                    │
               ▼                                    ▼
    ┌─────────────────────┐              ┌─────────────────────┐
    │  Image Registration │              │  Field Interpolator │
    │  ✅ EXISTS          │              │  ❌ TO BUILD        │
    │                     │              │                     │
    │  • Align frames     │              │  • scipy.interp     │
    │  • Correct drift    │              │  • Spatial grid     │
    │  • Edge detection   │              │  • Temporal sync    │
    └──────────┬──────────┘              └──────────┬──────────┘
               │                                    │
               ▼                                    │
    ┌─────────────────────┐                        │
    │  ROI Selection      │                        │
    │  ✅ EXISTS          │                        │
    │                     │                        │
    │  • Polygon mask     │                        │
    │  • Exclude artifacts│                        │
    └──────────┬──────────┘                        │
               │                                    │
               ▼                                    │
    ┌─────────────────────┐                        │
    │  Focus Loss Filter  │                        │
    │  ✅ EXISTS (NEW!)   │                        │
    │                     │                        │
    │  • Mark bad frames  │                        │
    │  • Auto exclude     │                        │
    └──────────┬──────────┘                        │
               │                                    │
               │                                    │
               ▼                                    │

╔═══════════════════════════════════════════════════════════════════════════════╗
║                         CELL SEGMENTATION PHASE                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝

    ┌─────────────────────────────────────────────────────────┐
    │  SegmentationService ✅ EXISTS (OPTIMIZED!)             │
    │                                                          │
    │  Model Options:                                          │
    │  • Cellpose (bact_phase_cp3, bact_fluor_cp3)            │
    │  • Omnipose (bact_phase_omni, bact_fluor_omni)          │
    │  • U-Net (custom trained)                                │
    │  • DeepBacs                                              │
    │                                                          │
    │  NEW Features:                                           │
    │  ✓ Frame-by-frame processing (memory optimized)         │
    │  ✓ Skip already processed frames                        │
    │  ✓ Progress tracking (tqdm)                             │
    │  ✓ Model-specific caching                               │
    └────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  Labeled Frames     │
                  │  (Cell Masks)       │
                  │                     │
                  │  Each pixel = Cell  │
                  │  label (0=bg, 1,2..)│
                  └─────┬──────┬────────┘
                        │      │
           ┌────────────┘      └────────────┐
           ▼                                 ▼

╔══════════════════════════════════╗  ╔══════════════════════════════════╗
║    MORPHOLOGY ANALYSIS           ║  ║    CELL TRACKING                 ║
╚══════════════════════════════════╝  ╚══════════════════════════════════╝

┌─────────────────────┐              ┌─────────────────────┐
│  Extract Metrics    │              │  BayesianTracker    │
│  ✅ EXISTS          │              │  ✅ EXISTS          │
│                     │              │                     │
│  Per frame:         │              │  • btrack engine    │
│  • Area             │              │  • Kalman filter    │
│  • Length/Width     │              │  • Object linking   │
│  • Aspect ratio     │              │  • Division detect  │
│  • Circularity      │              │                     │
│  • Solidity         │              │  Output:            │
│  • Orientation      │              │  • Persistent IDs   │
│                     │              │  • Trajectories     │
│  Store in:          │              │  • Lineage trees    │
│  MetricsService     │              │  • Parent-child     │
│  (Polars DataFrame) │              │                     │
└──────────┬──────────┘              └──────────┬──────────┘
           │                                    │
           ▼                                    ▼
┌─────────────────────┐              ┌─────────────────────┐
│  Classify Morphology│              │  Track Dictionary   │
│  ✅ EXISTS          │              │  ✅ EXISTS          │
│                     │              │                     │
│  Categories:        │              │  {                  │
│  • Healthy          │              │   ID: int           │
│  • Elongated        │              │   x: [...]          │
│  • Deformed         │              │   y: [...]          │
│  • Divided          │              │   t: [...]          │
│  • Artifact         │              │   parent: int       │
└──────────┬──────────┘              │   children: []      │
           │                         │  }                  │
           │                         └──────────┬──────────┘
           │                                    │
           │                                    │
           └────────────┬───────────────────────┘
                        │
                        ▼

╔═══════════════════════════════════════════════════════════════════════════════╗
║                    CRITICAL INTEGRATION POINT (TO COMPLETE!)                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝

                   ┌─────────────────────────────────┐
                   │  Cell ID → Morphology Mapping   │
                   │  ⚠️  PARTIAL (IN PROGRESS!)     │
                   │                                  │
                   │  Problem:                        │
                   │  • Metrics stored by (t,p)       │
                   │  • Tracks stored by cell_id      │
                   │  • Need to link them!            │
                   │                                  │
                   │  Solution:                       │
                   │  • Map segmentation labels       │
                   │  • Track cell_id through frames  │
                   │  • Join with MetricsService      │
                   └──────────────┬───────────────────┘
                                  │
                                  ▼

╔═══════════════════════════════════════════════════════════════════════════════╗
║                    CELL-BASED DATA REORGANIZATION                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝

                   ┌─────────────────────────────────┐
                   │  create_cell_based_dataset()    │
                   │  ✅ EXISTS (NEW!)               │
                   │                                  │
                   │  Merges:                         │
                   │  • Track trajectories            │
                   │  • Morphology time series        │
                   │  • Cell fate determination       │
                   │  • Lineage relationships         │
                   └──────────────┬───────────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────────┐
                   │  Cell Data Structure             │
                   │  ✅ EXISTS (NEW!)               │
                   │                                  │
                   │  cell_data[cell_id] = {          │
                   │    timepoints: [...]             │
                   │    x: [...], y: [...]            │
                   │    length: [...]                 │
                   │    width: [...]                  │
                   │    area: [...]                   │
                   │    states: [...]                 │
                   │    lifespan: int                 │
                   │    fate: str                     │
                   │    parent: int                   │
                   │    children: []                  │
                   │  }                               │
                   └──────────────┬───────────────────┘
                                  │
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼

╔══════════════════════════════╗        ╔══════════════════════════════╗
║  ENVIRONMENTAL MAPPING       ║        ║  FLUORESCENCE INTEGRATION    ║
║  ❌ TO BUILD (CRITICAL!)     ║        ║  ⚠️  TO VERIFY               ║
╚══════════════════════════════╝        ╚══════════════════════════════╝

┌─────────────────────────────┐        ┌──────────────────────────────┐
│  CellEnvironmentMapper      │        │  Fluorescence Quantification │
│  ❌ TO BUILD                │        │  ✅ EXISTS                   │
│                              │        │                              │
│  For each cell trajectory:  │        │  • Background subtraction    │
│                              │        │  • RPU calculations          │
│  1. Get (x,y,t) positions    │        │  • mCherry / YFP channels   │
│  2. Convert pixel → μm       │        │  • Per-cell extraction       │
│  3. Query field interpolator │        │                              │
│  4. Get velocity at position │        │  Add to cell_data:           │
│  5. Get pressure at position │        │  • fluo_mcherry: [...]       │
│  6. Get shear at position    │        │  • fluo_yfp: [...]           │
│                              │        │  • rpu_values: [...]         │
│  Add to cell_data:           │        └──────────────┬───────────────┘
│  • env_velocity: [...]       │                       │
│  • env_velocity_x: [...]     │                       │
│  • env_velocity_y: [...]     │                       │
│  • env_pressure: [...]       │                       │
│  • env_shear_stress: [...]   │                       │
└─────────────┬────────────────┘                       │
              │                                        │
              └─────────────────┬──────────────────────┘
                                │
                                ▼

╔═══════════════════════════════════════════════════════════════════════════════╗
║                    COMPLETE CELL VIEW DATA STRUCTURE                           ║
╚═══════════════════════════════════════════════════════════════════════════════╝

                   ┌─────────────────────────────────┐
                   │  FINAL CELL DATA (Target!)       │
                   │  🎯 GOAL                        │
                   │                                  │
                   │  cell_data[cell_id] = {          │
                   │                                  │
                   │    # Trajectory                  │
                   │    timepoints: [0,1,2,...]       │
                   │    x: [100, 102, ...]            │
                   │    y: [200, 198, ...]            │
                   │                                  │
                   │    # Morphology Time Series      │
                   │    length: [20.3, 21.1, ...]     │
                   │    width: [10.1, 10.5, ...]      │
                   │    area: [150, 155, ...]         │
                   │    aspect_ratio: [2.0, 2.1, ...] │
                   │    states: ["healthy", ...]      │
                   │                                  │
                   │    # Environmental Time Series   │
                   │    env_velocity: [0.5, 0.6, ...] │
                   │    env_pressure: [10, 11, ...]   │
                   │    env_shear: [0.1, 0.15, ...]   │
                   │                                  │
                   │    # Fluorescence (if available) │
                   │    fluo_mcherry: [100, 105, ...] │
                   │    fluo_yfp: [50, 52, ...]       │
                   │                                  │
                   │    # Cell Life History           │
                   │    lifespan: 45                  │
                   │    fate: "divided"               │
                   │    parent: 15                    │
                   │    children: [43, 44]            │
                   │  }                               │
                   └──────────────┬───────────────────┘
                                  │
                                  ▼

╔═══════════════════════════════════════════════════════════════════════════════╗
║                         ANALYSIS & CORRELATION                                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝

    ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
    │  Motility Analysis  │  │  Environmental      │  │  Correlation        │
    │  ✅ EXISTS          │  │  Analysis           │  │  Analysis           │
    │                     │  │  ⚠️  PARTIAL        │  │  ❌ TO BUILD        │
    │  • Velocity         │  │                     │  │                     │
    │  • Tortuosity       │  │  • Velocity profile │  │  • Morphology vs    │
    │  • Persistence      │  │  • Pressure zones   │  │    environment      │
    │  • Track length     │  │  • Shear patterns   │  │  • Motility vs      │
    │  • Region analysis  │  │  • Chamber regions  │  │    flow velocity    │
    │  • Division events  │  │                     │  │  • Cell fate vs     │
    │                     │  │  UI Ready:          │  │    conditions       │
    │  Output:            │  │  ✓ Env analysis tab │  │  • Generation vs    │
    │  • Per-cell metrics │  │  ✓ Filter options   │  │    environment      │
    │  • Population stats │  │  ✓ Visualization    │  │                     │
    └──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘
               │                        │                        │
               │                        │                        │
               └────────────────────────┴────────────┬───────────┘
                                                     │
                                                     ▼

╔═══════════════════════════════════════════════════════════════════════════════╗
║                         VISUALIZATION & EXPORT                                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Lineage Trees   │  │  Tracking Videos │  │  Density Animate │  │  Scatter Plots   │
│  ✅ EXISTS       │  │  ✅ EXISTS       │  │  ✅ EXISTS (NEW!)│  │  ✅ EXISTS       │
│                  │  │                  │  │                  │  │                  │
│  • Family trees  │  │  • GIF export    │  │  • Live mode     │  │  • Morphology    │
│  • Division viz  │  │  • MP4 export    │  │  • Cumulative    │  │  • Motility      │
│  • Color-coded   │  │  • Track overlay │  │  • Cell counter  │  │  • PCA plots     │
└──────────────────┘  └──────────────────┘  └──────────────────┘  └──────────────────┘

┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  CSV Export      │  │  Parquet Export  │  │  Population Plot │  │  Env Correlation │
│  ✅ EXISTS       │  │  ✅ EXISTS       │  │  ✅ EXISTS       │  │  ❌ TO BUILD     │
│                  │  │                  │  │                  │  │                  │
│  • Track data    │  │  • Metrics       │  │  • Fluorescence  │  │  • Scatter plots │
│  • Morphology    │  │  • Cell data     │  │  • Time series   │  │  • Heatmaps      │
│  • Per-cell      │  │  • Fast load     │  │  • Statistics    │  │  • Correlations  │
└──────────────────┘  └──────────────────┘  └──────────────────┘  └──────────────────┘

                                      │
                                      ▼

╔═══════════════════════════════════════════════════════════════════════════════╗
║                           PAPER DELIVERABLES                                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝

    ┌─────────────────────────────────────────────────────────────────┐
    │  Research Outputs (Paper Results)                                │
    │                                                                   │
    │  1. Cell morphology response to environmental conditions         │
    │     • How do cells elongate/deform under shear stress?          │
    │     • Does high velocity affect cell shape?                      │
    │                                                                   │
    │  2. Cell motility patterns in microfluidic flow                  │
    │     • Do cells move with or against flow?                        │
    │     • How does pressure affect motility?                         │
    │                                                                   │
    │  3. Division events and environmental factors                    │
    │     • Where do cells divide in the chamber?                      │
    │     • Does flow velocity affect division rate?                   │
    │                                                                   │
    │  4. Population dynamics in spatial gradients                     │
    │     • Cell distribution across velocity gradients                │
    │     • Survival rates in different pressure zones                 │
    │                                                                   │
    │  5. Lineage-specific environmental adaptation                    │
    │     • Do daughter cells behave differently than parents?         │
    │     • Generational responses to environmental stress             │
    │                                                                   │
    │  6. Fluorescence response to environmental conditions            │
    │     • Gene expression under flow/shear stress                    │
    │     • RPU correlation with environmental factors                 │
    └─────────────────────────────────────────────────────────────────┘
```

---

## Legend

- ✅ **EXISTS** - Fully implemented and working
- ✅ **EXISTS (NEW!)** - Recently added in partaker-2-Amby branch
- ⚠️ **PARTIAL** - Partially implemented, needs completion
- ⚠️ **TO VERIFY** - Exists but needs testing/validation
- ❌ **TO BUILD** - Not implemented, must be created
- 🎯 **GOAL** - Target data structure for paper analysis

---

## Critical Path to Paper Completion

### Phase 1: Complete Cell-Based Integration (Week 1-2)
1. ✅ Cell-based data structure (`cell_view_data.py`) - DONE
2. ⚠️ Cell ID to morphology mapping - **IN PROGRESS**
3. ⚠️ Fluorescence integration verification - **TO TEST**

### Phase 2: Environmental Data Integration (Week 2-4)
1. ❌ Build COMSOL data loader
2. ❌ Implement field interpolator
3. ❌ Create cell-environment mapper
4. ❌ Extend cell_data with environmental time series

### Phase 3: Analysis & Paper Results (Week 4-6)
1. ❌ Correlation analysis module
2. ❌ Statistical tests (morphology vs environment)
3. ❌ Environmental visualization overlays
4. ❌ Generate paper figures and datasets

---

## Key Paper Questions This Workflow Answers

1. **How do bacterial cells respond morphologically to microfluidic flow?**
   - Cell length/width changes vs velocity
   - Deformation under shear stress
   - Morphological state distribution across pressure zones

2. **What is the relationship between cell motility and environmental flow?**
   - Cell velocity vs fluid velocity
   - Movement patterns in velocity gradients
   - Chemotaxis vs flow-driven displacement

3. **Do environmental conditions affect cell division?**
   - Division frequency in high/low flow regions
   - Division events vs shear stress
   - Spatial distribution of division events

4. **How do different generations adapt to environmental stress?**
   - Parent vs daughter cell morphology
   - Generational motility differences
   - Lineage-specific survival in extreme conditions

5. **Does gene expression correlate with environmental conditions?**
   - Fluorescence (promoter activity) vs velocity
   - RPU values across pressure gradients
   - Stress response activation zones

---

## Data Flow Summary

```
Microscopy + COMSOL → Load & Preprocess → Segment → Track + Morphology
→ Link Cell IDs → Cell-Based Dataset → Add Environment → Complete Cell Data
→ Correlation Analysis → Paper Results
```

**Current Progress: ~70% Complete**
- ✅ Microscopy pipeline: 100%
- ✅ Tracking & morphology: 95%
- ⚠️ Cell integration: 70%
- ❌ Environmental data: 0%
- ❌ Correlation analysis: 0%
