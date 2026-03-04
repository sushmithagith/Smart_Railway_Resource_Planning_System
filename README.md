# Smart Railway Resource Planning System 🚆

A web-based data analytics and forecasting dashboard designed to help railway planners make data-driven decisions for train resource allocation.

## Features

- **Upload Data:** Load historical operational data via CSV or Excel directly into the app.
- **Data Visualizations:** Interactive Dashboards utilizing Plotly to show passenger demand over time, busy routes, and platform constraints.
- **Overcrowding Alerts:** Automatically flags and highlights services where passenger occupancy exceeds total capacity.
- **Demand Prediction (Machine Learning):** Utilizes a Scikit-Learn `RandomForestRegressor` to forecast exact passenger requirements based on route, time, and external factors like weekends/holidays.
- **Active Resource Allocation:** Translates ML forecasts into plain-english dynamic recommendations—telling planners exactly how many additional coaches to attach, or when to schedule relief trains.

## Setup & Installation

**Prerequisites:** Python 3.8+ installed on your machine.

1. **Clone or Download** this project folder to your local machine.
2. **Open a Terminal** (or command prompt) and navigate to the project directory:
   ```bash
   cd "path/to/Smart Railway Resource Planning System"
   ```
3. **Install Dependencies.** It is highly recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

## Generating Mock Data

If you don't have historical data on hand, you can generate a realistic dataset to test the dashboard.

Run the provided script:
```bash
python generate_mock_data.py
```
This will create a `railway_data.csv` file in your directory containing 2,000 randomized records of historical train journeys.

## Starting the Dashboard

Launch the application using the Streamlit CLI:
```bash
streamlit run app.py
```
This will automatically open the dashboard in your default web browser (typically at `http://localhost:8501`).

### How to Use
1. Once launched, look at the sidebar on the left and click **Browse Files**.
2. Upload the `railway_data.csv` file you generated (or your own structured CSV).
3. Navigate between the top tabs:
   * **Overview:** System-level KPIs and critical overcrowding detections.
   * **Visualizations:** Drill down into interactive route and hour-of-the-day heatmaps.
   * **Demand Prediction:** Enter hypothetical scenarios (e.g. 5:00 PM on a Holiday) to generate a passenger volume forecast.
   * **Resource Recommendations:** After making a prediction, check this tab to receive automated instructions on how to scale capacity (coaches, platforms, staffing) for that specific run!

## Technology Stack
- **Dashboard Framework:** Streamlit
- **Data Manipulation:** Pandas & NumPy
- **Visualizations:** Plotly Express
- **Machine Learning:** Scikit-Learn (Random Forest)
