# Data Center Material Forecasting Solution

A comprehensive predictive analytics solution for material requirement prediction in supply chain management, specifically designed for a 25MW Data Center project in Navi Mumbai with a budget of ₹1,875 Cr.

## 🎯 Project Overview

This solution addresses the complete material forecasting lifecycle for construction projects, from predictive modeling to vendor identification and procurement management. It's specifically tailored for the data center construction project with the following specifications:

- **Project Type**: 25MW Data Center
- **Location**: Navi Mumbai, Maharashtra
- **Built-up Area**: 200,000 sq ft
- **Project Volume**: ₹1,875 Cr
- **Timeline**: 2 years construction period

## 🏗️ Solution Components

### Stage 1: Predictive Model (50 Points)
- **model_training.py** - Advanced ML model with feature engineering
- Multiple algorithms (RandomForest, GradientBoosting, LinearRegression)
- Feature importance analysis and model explainability
- Comprehensive material predictions for data center

### Stage 2: Local Host Application (5 Points)
- **dashboard.py** - Advanced Streamlit dashboard with chatbot
- Interactive visualizations and real-time predictions
- AI-powered chatbot for project queries
- Multi-page navigation with comprehensive analytics

### Stage 3: Vendor Identification (15 Points)
- **vendor_scraper.py** - Multi-source vendor identification
- Google Search, IndiaMart, and TradeIndia integration
- Vendor scoring and recommendation system
- Contact information and rating analysis

### Stage 4: Construction Project Schedule (10 Points)
- **construction_schedule_procurement.py** - Integrated project scheduling
- 9-phase construction timeline with critical path analysis
- Material delivery scheduling with lead times
- Resource requirement planning

### Stage 5: Procurement Management (10 Points)
- **procurement_management_platform.py** - Complete procurement platform
- Workflow management and vendor selection
- Cost management and budget tracking
- Quality management and reporting

### Stage 6: Integration & Deployment (10 Points)
- Comprehensive data integration across all modules
- Automated report generation
- Risk assessment and mitigation strategies
- Complete documentation and deployment guide

## 🚀 Quick Start

### Prerequisites

```bash
# Install Python 3.8+ and required packages
pip install -r requirements.txt
```

### Step 1: Train the Model

```bash
python model_training.py
```

**Outputs:**
- `material_model.pkl` - Trained model with performance metrics
- `data_center_predictions.csv` - Material predictions for data center
- Model performance metrics (RMSE, MAPE, R²)

### Step 2: Identify Vendors

```bash
python vendor_scraper.py
```

**Outputs:**
- `vendor_recommendations.csv` - Verified vendor recommendations
- Contact details and recommendation scores
- Multi-source vendor data

### Step 3: Create Project Schedule

```bash
python construction_schedule_procurement.py
```

**Outputs:**
- `construction_phases.csv` - Detailed construction phases
- `material_timeline.csv` - Material delivery timeline
- `procurement_schedule.csv` - Procurement milestones
- `monthly_procurement_costs.csv` - Cost distribution

### Step 4: Launch Dashboard

```bash
streamlit run dashboard.py
```

**Features:**
- Interactive material predictions
- Vendor recommendations with filtering
- Procurement timeline visualization
- AI chatbot for project queries
- Model performance analytics

### Step 5: Access Procurement Platform

```bash
streamlit run procurement_management_platform.py
```

**Features:**
- Complete procurement workflow management
- Vendor selection and evaluation
- Cost management and budget tracking
- Quality management standards
- Comprehensive reporting and analytics

## 📊 Key Features

### Predictive Analytics
- **Multi-algorithm approach** with automatic best model selection
- **Feature engineering** with project density and scale metrics
- **Model explainability** with feature importance analysis
- **Cost estimation** with unit cost predictions

### Vendor Management
- **Multi-source vendor identification** (Google, IndiaMart, TradeIndia)
- **Vendor scoring system** based on reliability and experience
- **Contact information extraction** with ratings and reviews
- **Recommendation engine** for optimal vendor selection

### Project Scheduling
- **9-phase construction timeline** with critical path analysis
- **Material delivery scheduling** with lead time optimization
- **Resource requirement planning** with cost distribution
- **Risk assessment** with milestone tracking

### Procurement Management
- **Complete workflow management** from planning to payment
- **Vendor selection interface** with comparison tools
- **Cost management** with budget tracking and optimization
- **Quality management** with standards compliance
- **Reporting and analytics** with comprehensive dashboards

## 📈 Model Performance

The model achieves:
- **RMSE**: Optimized across multiple algorithms
- **MAPE**: Minimized prediction errors
- **R² Score**: High correlation with actual requirements
- **Feature Importance**: Clear interpretability for decision making

## 🎯 Data Center Specific Predictions

The solution provides detailed predictions for:
- Medium Voltage Switchgear
- Transformers (2-3MVA)
- Chillers/CRAHs/CRACs
- Cement and Bricks
- Steel Reinforcement
- UPS Systems
- Generator Sets
- Fire Suppression Systems
- HVAC Ductwork

## 🔧 Customization

### For Different Project Types
1. Modify material lists in `model_training.py`
2. Update lead times in `construction_schedule_procurement.py`
3. Adjust vendor search terms in `vendor_scraper.py`
4. Customize dashboard parameters in `dashboard.py`

### For Different Locations
1. Update location parameters in vendor scraper
2. Modify regional features in model training
3. Adjust lead times based on local suppliers
4. Update cost estimates for regional pricing

## 📋 Output Files

The solution generates comprehensive outputs:
- **Model files**: `material_model.pkl`
- **Predictions**: `data_center_predictions.csv`
- **Vendors**: `vendor_recommendations.csv`
- **Schedules**: `construction_phases.csv`, `material_timeline.csv`, `procurement_schedule.csv`
- **Costs**: `monthly_procurement_costs.csv`

## 🌐 Deployment

### Local Deployment
```bash
# Run individual components
python model_training.py
python vendor_scraper.py
python construction_schedule_procurement.py

# Launch web applications
streamlit run dashboard.py
streamlit run procurement_management_platform.py
```
