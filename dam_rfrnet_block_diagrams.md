# System Block Diagram: DAM-RFRNet Architecture (Corrected Connections)

This document presents the corrected System Block Diagram for **DAM-RFRNet Architecture**:
- **Fixed Connection**: The output of `Feature Extraction` connects **ONLY** to `Integration with Original Image` (stray arrow to `Final Dehazed Output` removed).

---

## 1. Corrected System Flowchart

```
                                  +-----------------------+
                                  |    Preprocessing      |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  | Down-sampling (x1/4)  |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  | Coarse K-Value        |
                                  | Estimation            |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  |  Up-sampling (x4)     |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  |  Feature Extraction   |
                                  +-----------+-----------+
                                              |
+-------------------+                         v (ONLY ONE ARROW)
|   RGB Hazy Image  |-------------> +-------------------------------+
|     I(x)          |               | Integration with Original     |
+-------------------+               | Image                         |
                                    +---------------+---------------+
                                                    |
                                                    v
                                    +-------------------------------+
                                    |  NOVEL BLOCK 1:               |
                                    |  Residual Feature Refinement  |
                                    |  Block (RFRB)                 |
                                    +---------------+---------------+
                                                    |
                                                    v
                                    +-------------------------------+
                                    |  NOVEL BLOCK 2:               |
                                    |  Dual Spatial-Channel         |
                                    |  Attention Mechanism (DSCA)   |
                                    +---------------+---------------+
                                                    |
                                                    v
                                    +-------------------------------+
                                    |  K-Value Estimation           |
                                    +---------------+---------------+
                                                    |
                                                    v
                                    +-------------------------------+
                                    |  Reconstruction               |
                                    |  (using physical model)       |
                                    +---------------+---------------+
                                                    |
                                                    v
                                    +-------------------------------+
                                    |  Final Dehazed Output J(x)    |
                                    +-------------------------------+
```

---

## 2. Interactive Mermaid Flowchart

```mermaid
flowchart TD
    classDef input fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef novel fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px;
    classDef block fill:#ffffff,stroke:#616161,stroke-width:1.5px;
    classDef output fill:#fff3e0,stroke:#f57c00,stroke-width:2px;

    RGB["📷 RGB Hazy Image\n(Input Photo)"]:::input --> Pre["Preprocessing"]:::block
    Pre --> Down["Down-sampling (x1/4)"]:::block
    Down --> CoarseK["Coarse K-Value Estimation"]:::block
    CoarseK --> Up["Up-sampling (x4)"]:::block
    Up --> Feat["Feature Extraction"]:::block
    
    Feat --> Integration["Integration with Original Image"]:::block
    RGB -- Direct Skip Path --> Integration

    Integration --> RFRB["NOVEL BLOCK 1:\nResidual Feature Refinement Block (RFRB)"]:::novel
    RFRB --> DSCA["NOVEL BLOCK 2:\nDual Spatial-Channel Attention Mechanism (DSCA)"]:::novel
    
    DSCA --> MainK["K-Value Estimation"]:::block
    MainK --> Recon["Reconstruction"]:::block
    Recon --> FinalOut["🖼️ Final Dehazed Output\n(Output Photo)"]:::output
```

---

## 3. Summary of Fixes
1. **Removed Stray Arrow**: `Feature Extraction` output arrow points **exclusively** down into `Integration with Original Image`.
2. **Correct Processing Path**: Ensures features are integrated before being passed to **RFRB** and **DSCA**.
