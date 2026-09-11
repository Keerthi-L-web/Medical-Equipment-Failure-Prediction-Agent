# Exploratory Data Analysis & Schema Audit Report

## 1. Dataset Overview

- **Devices Total**: `118,249` rows
- **Events Total**: `124,969` rows
- **Manufacturers Total**: `31,827` rows
- **Devices with Recorded Events**: `118,249` (100.00%)

## 2. Key Column Distributions

### Risk Class (Devices)
- `nan`: 85,301
- `2`: 24,693
- `1`: 5,654
- `3`: 2,255
- `Unclassified`: 242
- `Not Classified`: 86
- `HDE`: 16
- `II`: 2

### Event Types & Action Summaries
- `nan`: 95,864
- `Recall`: 2,137
- `Recall for Product Correction`: 1,999
- `Correction`: 1,576
- `Recall/Exchange`: 1,339
- `Upgrade software`: 693
- `Notification made by the company in compliance with RDC 23/2012 (which provides for the obligation of execution and notification of field action by the holder of the registration of the product for health)`: 569
- `Change instructions`: 465
- `Change products`: 408
- `Other`: 295

## 3. Date Ranges & Temporal Cutoff Assessment

- **Earliest Event Date**: `0011-05-06 00:00:00`
- **Latest Event Date**: `2200-02-15 00:00:00`
- **Recommended Temporal Cutoff**: `2016-01-01` (or 75th percentile date)
- **Prediction Horizon**: 12 Months post-cutoff

## 4. Key Takeaways & ML Pipeline Recommendations

1. **Relational Linkage**: Devices link to Manufacturers via `manufacturer_id` and Events link to Devices via `device_id`.
2. **Device Categorization**: Map raw `name` & `classification` into major clinical types (Monitoring, Respiratory, Pumps, Imaging, Diagnostic, Surgical, Implantable, Other).
3. **Target Imbalance**: Out of all devices, a fraction experience events in the future prediction window. XGBoost `scale_pos_weight` will handle class imbalance.
4. **Synthetic Telemetry Requirement**: Synthetic telemetry will be generated for monitored devices with realistic health degradation curves.
