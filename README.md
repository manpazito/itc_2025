# Interdistrict Transportation Competition 2025 — UC Berkeley

**Repository:** `itc_2025`
**Access:** Restricted to UC Berkeley Cal Transpo student teams until the conclusion of ITC 2025.

---

## Overview

This repository supports UC Berkeley’s participation in the **Interdistrict Transportation Competition (ITC) 2025**, hosted by **UC Davis** from November 14–16, 2025.

The competition focuses on improving multimodal operations along **2nd Street in Downtown Davis**, connecting to the Davis Amtrak Station.

Project goals include:

* Enhancing safety and connectivity for pedestrians and cyclists
* Improving bus circulation and ridership
* Supporting transit-oriented and mixed-use development
* Proposing data-driven multimodal design and policy strategies

This repository serves as the central workspace for all **analysis, modeling, and code development** related to the project.
Developed by the **UC Berkeley Cal Transpo Consulting Subgroup**:
**Dan-Vy Nguyen, Elaine Shim, Elena [Last Name], Ivy Luu, Jack Williams, and Manuel Martinez Garcia.**

---

## Components

### 1. Initial Analysis

Contains Python notebooks and scripts for:

* Data cleaning and exploratory analysis
* Crash data mapping (TIMS) and area of interest (AOI) filtering
* Multimodal network and safety visualization

Key libraries: `geopandas`, `pandas`, `matplotlib`, `shapely`, and `contextily`.

### 2. Traffic Modeling

Includes **PTV Vissim** files and configuration data for:

* Baseline network and intersection control
* Scenario testing and traffic performance modeling
* Signal timing and multimodal priority evaluation

### 3. Documentation

Includes the official **ITC 2025 prompt**, this `README.md`, and supporting notes or figures for deliverables.

---

## Deliverables

Each team must produce:

1. **Project Report (PDF)** — including Overview, Existing Conditions, Design Features, and Design Analysis
2. **Presentation Slideshow (.pptx)** — 8-minute summary for the ITC presentation

Outputs from this repository support both deliverables.

---

## Setup and Usage

**1. Clone repository**

```bash
git clone https://github.com/<your-org>/itc_2025.git
cd itc_2025
```

**2. Create environment**

```bash
conda create -n itc_2025 python=3.11 geopandas matplotlib seaborn numpy pandas shapely contextily
conda activate itc_2025
```

**3. Run analysis**
Open and execute:

```
init_analysis/initial_analysis.ipynb
```

Outputs and figures will be generated automatically in the corresponding folders.

---

## Access Policy

This repository is **restricted to UC Berkeley Cal Transpo student teams** participating in ITC 2025.
Access for other institutions will be granted **after the competition concludes**.
Reproduction or distribution of materials prior to that date is not permitted.

---

## Acknowledgments

* Interdistrict Transportation Competition (ITC) 2025 — UC Davis
* UC Berkeley Institute of Transportation Engineers (ITE)
* UC Berkeley Cal Transpo Team

## License

This repository is distributed under a **Restricted Academic License**.  
Use is limited to UC Berkeley Cal Transpo student teams participating in ITC 2025.  
See the [LICENSE](./LICENSE) file for full details.
