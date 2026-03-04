import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_mock_data(num_records=2000):
    # Base data configurations
    routes = [
        "New York - Washington DC", 
        "London - Paris", 
        "Tokyo - Osaka", 
        "Mumbai - Delhi", 
        "Sydney - Melbourne",
        "Berlin - Munich",
        "Toronto - Montreal",
        "Chicago - St. Louis",
        "Beijing - Shanghai",
        "Los Angeles - San Francisco"
    ]
    
    train_ids = [f"TRN-{str(i).zfill(3)}" for i in range(1, 21)]
    
    # Generate dates over the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    data = []
    
    for _ in range(num_records):
        train_id = random.choice(train_ids)
        route = random.choice(routes)
        
        # Random date and time within the last 30 days
        random_seconds = random.randint(0, int((end_date - start_date).total_seconds()))
        dt = start_date + timedelta(seconds=random_seconds)
        
        # Determine if weekend
        is_weekend = dt.weekday() >= 5
        
        # Determine if simulated holiday (random 10% chance)
        is_holiday = random.random() < 0.1
        
        holiday_or_weekend = "Yes" if is_weekend or is_holiday else "No"
        
        # Based on time, route, and weekend/holiday, generate realistic numbers
        # Peak hours are typically 7-9 AM and 5-7 PM
        is_peak_hour = (7 <= dt.hour <= 9) or (17 <= dt.hour <= 19)
        
        base_demand = random.randint(100, 300)
        if is_peak_hour:
            base_demand += random.randint(150, 300)
        if holiday_or_weekend == "Yes":
            base_demand += random.randint(100, 250)
            
        passenger_count = base_demand
        
        # Define coaches (assume each coach holds ~60 people max)
        # Randomly assign coaches from 5 to 15
        num_coaches = random.randint(5, 15)
        total_capacity = num_coaches * 60
        
        # Seat occupancy percentage
        seat_occupancy = round((passenger_count / total_capacity) * 100, 2)
        
        platform_number = random.randint(1, 12)
        
        # Delays in minutes (mostly 0, but sometimes up to 120 minutes)
        delay_minutes = 0
        if random.random() < 0.2: # 20% chance of delay
            delay_minutes = random.randint(5, 120)
            
        data.append({
            "Train ID": train_id,
            "Route": route,
            "Date and Time": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Passenger Count": passenger_count,
            "Number of Coaches": num_coaches,
            "Total Capacity": total_capacity,
            "Seat Occupancy (%)": seat_occupancy,
            "Platform Number": platform_number,
            "Delay Records (mins)": delay_minutes,
            "Holiday or Weekend Indicator": holiday_or_weekend
        })

    df = pd.DataFrame(data)
    df = df.sort_values(by="Date and Time").reset_index(drop=True)
    df.to_csv("railway_data.csv", index=False)
    print("Successfully generated 'railway_data.csv' with", num_records, "records.")

if __name__ == "__main__":
    generate_mock_data()
