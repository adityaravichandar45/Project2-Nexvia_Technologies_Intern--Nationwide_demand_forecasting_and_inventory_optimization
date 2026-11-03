# Project2-Nexvia_Technologies_Intern-Nationwide_demand_forecasting_and_inventory_optimization

# Nationwide Demand Forecasting & Inventory Optimization System

An end-to-end nationwide demand forecasting and inventory optimization pipeline built for the M5 dataset. This project integrates machine learning time-series modeling (LightGBM), bottom-up hierarchical reconciliation, uncertainty intervals, mathematical inventory optimization (Google OR-Tools), and an interactive web dashboard deployed live via Render and PlotlyCloud.

---

## Live Demo
* **Deployed Web Application:1) Access Live App on Render- https://project2-nexvia-technologies-intern.onrender.com/
                             2) Access Live App on PlotlyCloud -https://dfioadi.plotly.app/

---

##  System Architecture & Data Flow
1. **Data Ingestion & Processing:** Historical retail sales data is processed with time-series features (lags, rolling statistics, calendar events).
2. **Hierarchical Reconciliation:** Granular item-store predictions are bottom-up aggregated using Pandas to ensure strict mathematical consistency with upper store and state totals.
3. **Machine Learning Forecasting:** A trained gradient boosting regressor (`lgb_dfio_model.pkl`) predicts future item-level demand.
4. **Inventory Optimization:** Google OR-Tools computes optimal order quantities and safety stock levels by balancing holding costs against stockout penalties under capacity constraints.
5. **Dashboard Deployment:** A Plotly Dash application presents interactive time-series curves, accuracy performance metrics, and inventory recommendations.

---

##  Performance & Accuracy Metrics
* **WAPE (Weighted Absolute Percentage Error):** 5.24%
* **RMSSE (Root Mean Squared Scaled Error):** 0.290

---

##  Tech Stack
* **Machine Learning:** LightGBM, Joblib
* **Optimization:** Google OR-Tools
* **Data Processing:** Pandas, NumPy
* **Web Dashboard:** Plotly Dash, HTML/CSS
* **Deployment & Hosting:** PlotlyCloud, Render, Gunicorn

---


