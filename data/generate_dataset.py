"""
=========================================================
Honeywell Grade Change Intelligence
Synthetic Dataset Generator
PART 1
=========================================================
"""
def generate_dataset():


    import os
    import random
    from datetime import datetime, timedelta

    import numpy as np
    import pandas as pd

    # ==========================================================
    # RANDOM SEED
    # ==========================================================

    np.random.seed(42)
    random.seed(42)

    # ==========================================================
    # OUTPUT
    # ==========================================================

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    OUTPUT_DIR = os.path.join(BASE_DIR, "synthetic")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    OUTPUT_FILE = os.path.join(
        OUTPUT_DIR,
        "paper_grade_dataset.csv"
    )

    # ==========================================================
    # SIMULATION SETTINGS
    # ==========================================================

    NUM_TRANSITIONS = 120

    TIMESTEPS = 180

    TIME_INTERVAL = 1

    FAILURE_RATE = 0.30

    # ==========================================================
    # PAPER RECIPES
    # ==========================================================

   # ==========================================================
# PAPER RECIPES
# ==========================================================

    RECIPES = {

        "42_70": {

            "recipe_id": "R001",

            "from": "42 GSM",

            "to": "70 GSM",

            "target_bw": 70,

            "stock_start": 1200,

            "stock_end": 1750,

            "steam_start": 210,

            "steam_end": 290,

            "speed_start": 850,

            "speed_end": 720,

            "filler_start": 150,

            "filler_end": 220

        },

        "70_90": {

            "recipe_id": "R002",

            "from": "70 GSM",

            "to": "90 GSM",

            "target_bw": 90,

            "stock_start": 1750,

            "stock_end": 1480,

            "steam_start": 290,

            "steam_end": 240,

            "speed_start": 720,

            "speed_end": 800,

            "filler_start": 220,

            "filler_end": 180

        },

        "90_120": {

            "recipe_id": "R003",

            "from": "90 GSM",

            "to": "120 GSM",

            "target_bw": 120,

            "stock_start": 1480,

            "stock_end": 1900,

            "steam_start": 240,

            "steam_end": 310,

            "speed_start": 800,

            "speed_end": 650,

            "filler_start": 180,

            "filler_end": 250

        },

        "120_42": {

            "recipe_id": "R004",

            "from": "120 GSM",

            "to": "42 GSM",

            "target_bw": 42,

            "stock_start": 1900,

            "stock_end": 1200,

            "steam_start": 310,

            "steam_end": 210,

            "speed_start": 650,

            "speed_end": 850,

            "filler_start": 250,

            "filler_end": 150

        }

    }

    # ==========================================================
    # MACHINE LIMITS
    # ==========================================================

    LIMITS = {

    "stock":(900,2000),

    "steam":(150,320),

    "speed":(600,1000),

    "filler":(100,260)

    }

    # ==========================================================
    # DISTURBANCES
    # ==========================================================

    DISTURBANCES = [

    "None",

    "Steam Drop",

    "Pump Oscillation",

    "Valve Delay",

    "Stock Surge",

    "Sensor Noise"

    ]

    # ==========================================================
    # SMOOTH TRANSITION
    # ==========================================================

    def s_curve(t,start,end):

        return start+(end-start)*(3*t**2-2*t**3)

    # ==========================================================
    # SENSOR NOISE
    # ==========================================================

    def noise(value,sigma):

        return value+np.random.normal(0,sigma)

    # ==========================================================
    # CLIP
    # ==========================================================

    def clip(value,low,high):

        return max(low,min(value,high))

    # ==========================================================
    # DISTURBANCE
    # ==========================================================

    def generate_disturbance():

        if random.random()<FAILURE_RATE:

            return random.choice(DISTURBANCES[1:])

        return "None"

    # ==========================================================
    # PROCESS PHASE
    # ==========================================================

    def process_phase(step):

        ratio=step/TIMESTEPS

        if ratio<0.2:

            return "Ramp Up"

        elif ratio<0.7:

            return "Transition"

        elif ratio<0.9:

            return "Stabilization"

        return "Steady State"

    # ==========================================================
    # APPLY DISTURBANCE
    # ==========================================================

    def apply_disturbance(stock,steam,speed,filler,disturbance):

        if disturbance=="Steam Drop":

            steam-=15

        elif disturbance=="Pump Oscillation":

            stock-=50

        elif disturbance=="Valve Delay":

            filler+=10

        elif disturbance=="Stock Surge":

            stock+=70

        elif disturbance=="Sensor Noise":

            speed+=20

        stock=clip(stock,*LIMITS["stock"])
        steam=clip(steam,*LIMITS["steam"])
        speed=clip(speed,*LIMITS["speed"])
        filler=clip(filler,*LIMITS["filler"])

        return stock,steam,speed,filler

    # ==========================================================
    # DIGITAL TWIN
    # ==========================================================

    class PaperMachine:

        def __init__(self):

            self.prev_bw=None
            self.prev_moisture=None
            self.prev_ash=None
            self.prev_caliper=None

        def headbox_pressure(self,stock,speed):

            return 0.12*stock-0.03*(speed/10)+np.random.normal(0,0.5)

        def basis_weight(self,stock,speed,headbox):

            bw=(stock/speed)*56.5

            bw+=(headbox-120)*0.05

            if self.prev_bw is not None:

                bw=0.65*self.prev_bw+0.35*bw

            bw+=np.random.normal(0,0.4)

            self.prev_bw=bw

            return bw
        # ==========================================================
    # QUALITY MODELS
    # ==========================================================

        def moisture(self, steam, bw):

            value = 7.5 - 0.0105 * steam + 0.020 * bw

            if self.prev_moisture is not None:

                value = 0.70 * self.prev_moisture + 0.30 * value

            value += np.random.normal(0, 0.08)

            self.prev_moisture = value

            return value

        def ash(self, filler, stock):

            value = (filler / stock) * 85

            if self.prev_ash is not None:

                value = 0.60 * self.prev_ash + 0.40 * value

            value += np.random.normal(0, 0.10)

            self.prev_ash = value

            return value

        def caliper(self, bw, moisture):

            value = 0.82 * bw + 0.25 * moisture

            if self.prev_caliper is not None:

                value = 0.75 * self.prev_caliper + 0.25 * value

            value += np.random.normal(0, 0.25)

            self.prev_caliper = value

            return value

    # ==========================================================
    # CONTROLLER
    # ==========================================================

    def controller(current, target, max_step):

        error = target - current

        if abs(error) <= max_step:

            return target

        if error > 0:

            return current + max_step

        return current - max_step

    # ==========================================================
    # OFF SPEC
    # ==========================================================

    def off_spec(bw, target):

        return int(abs(bw - target) > target * 0.025)

    # ==========================================================
    # ALARM ENGINE
    # ==========================================================

    def alarm(bw, moisture, ash, target):

        upper = target * 1.025

        lower = target * 0.975

        if bw > upper:

            return "Basis Weight High"

        if bw < lower:

            return "Basis Weight Low"

        if moisture > 7.5:

            return "High Moisture"

        if ash > 15:

            return "High Ash"

        return "Normal"

    # ==========================================================
    # OPERATOR ACTION
    # ==========================================================

    def operator_action(bw, moisture, ash, target):

        upper = target * 1.025

        lower = target * 0.975

        if bw > upper:

            return "Reduce Stock Flow"

        if bw < lower:

            return "Increase Stock Flow"

        if moisture > 7.5:

            return "Increase Steam"

        if ash > 15:

            return "Reduce Filler"

        return "Maintain"

    # ==========================================================
    # RECOMMENDATION SOURCE
    # ==========================================================

    def recommendation_source(action):

        if action == "Maintain":

            return "Recipe Rules"

        return random.choice(

            [

                "Historical Data",

                "Correlation Engine",

                "ML Prediction"

            ]

        )

    # ==========================================================
    # STABILIZATION TIME
    # ==========================================================

    def stabilization_time(disturbance):

        if disturbance == "None":

            return random.randint(8,15)

        if disturbance == "Steam Drop":

            return random.randint(20,30)

        if disturbance == "Pump Oscillation":

            return random.randint(18,28)

        if disturbance == "Valve Delay":

            return random.randint(15,25)

        if disturbance == "Stock Surge":

            return random.randint(18,26)

        return random.randint(10,20)

    # ==========================================================
    # PROCESS SIMULATION
    # ==========================================================

    def simulate(recipe, step, machine):

        t = step / (TIMESTEPS - 1)

        stock = s_curve(
            t,
            recipe["stock_start"],
            recipe["stock_end"]
        )

        steam = s_curve(
            t,
            recipe["steam_start"],
            recipe["steam_end"]
        )

        speed = s_curve(
            t,
            recipe["speed_start"],
            recipe["speed_end"]
        )

        filler = s_curve(
            t,
            recipe["filler_start"],
            recipe["filler_end"]
        )

        stock = noise(stock,5)
        steam = noise(steam,1)
        speed = noise(speed,2)
        filler = noise(filler,2)

        stock = controller(stock, recipe["stock_end"], 12)
        steam = controller(steam, recipe["steam_end"], 3)
        speed = controller(speed, recipe["speed_end"], 5)
        filler = controller(filler, recipe["filler_end"], 4)

        disturbance = generate_disturbance()

        stock, steam, speed, filler = apply_disturbance(
            stock,
            steam,
            speed,
            filler,
            disturbance
        )

        headbox = machine.headbox_pressure(stock, speed)

        bw = machine.basis_weight(stock, speed, headbox)

        moisture = machine.moisture(steam, bw)

        ash = machine.ash(filler, stock)

        caliper = machine.caliper(bw, moisture)

        return {

            "Stock_Flow": round(stock,2),

            "Steam_Pressure": round(steam,2),

            "Machine_Speed": round(speed,2),

            "Filler_Flow": round(filler,2),

            "Headbox_Pressure": round(headbox,2),

            "Basis_Weight": round(bw,2),

            "Moisture": round(moisture,2),

            "Ash": round(ash,2),

            "Caliper": round(caliper,2),

            "Disturbance": disturbance

        }
    # ==========================================================
    # DATASET GENERATION
    # ==========================================================

    dataset = []

    machine = PaperMachine()

    current_time = datetime(2026, 1, 1, 8, 0)

    transition_id = 1

    for _ in range(NUM_TRANSITIONS):

        recipe_name = random.choice(list(RECIPES.keys()))

        recipe = RECIPES[recipe_name]

        for step in range(TIMESTEPS):

            process = simulate(recipe, step, machine)

            target_bw = recipe["target_bw"]

            upper = round(target_bw * 1.025, 2)

            lower = round(target_bw * 0.975, 2)

            current_offspec = off_spec(
                process["Basis_Weight"],
                target_bw
            )

            action = operator_action(
                process["Basis_Weight"],
                process["Moisture"],
                process["Ash"],
                target_bw
            )

            dataset.append({

                "Timestamp": current_time,

                "Transition_ID": f"T{transition_id:03d}",

                "Recipe_ID": recipe["recipe_id"],

                "From_Grade": recipe["from"],

                "To_Grade": recipe["to"],

                "Phase": process_phase(step),

                "Step": step,

                "Stock_Flow": process["Stock_Flow"],

                "Filler_Flow": process["Filler_Flow"],

                "Steam_Pressure": process["Steam_Pressure"],

                "Machine_Speed": process["Machine_Speed"],

                "Headbox_Pressure": process["Headbox_Pressure"],

                "Basis_Weight": process["Basis_Weight"],

                "Target_BW": target_bw,

                "BW_Upper_Limit": upper,

                "BW_Lower_Limit": lower,

                "Moisture": process["Moisture"],

                "Ash": process["Ash"],

                "Caliper": process["Caliper"],

                "Alarm": alarm(
                    process["Basis_Weight"],
                    process["Moisture"],
                    process["Ash"],
                    target_bw
                ),

                "Operator_Action": action,

                "Suggested_Setpoint": action,

                "Recommendation_Source":
                recommendation_source(action),

                "Scenario":
                "Failure"
                if process["Disturbance"] != "None"
                else "Optimal",

                "Disturbance":
                process["Disturbance"],

                "Stabilization_Time":
                stabilization_time(
                    process["Disturbance"]
                ),

                "Off_Spec":
                current_offspec

            })

            current_time += timedelta(minutes=1)

        transition_id += 1

        current_time += timedelta(minutes=30)

    # ==========================================================
    # CREATE DATAFRAME
    # ==========================================================

    df = pd.DataFrame(dataset)

    # ==========================================================
    # FUTURE TARGETS
    # ==========================================================

    forecast_horizon = 5

    df["Future_Basis_Weight"] = (

        df.groupby("Transition_ID")

        ["Basis_Weight"]

        .shift(-forecast_horizon)

    )

    df["Future_Moisture"] = (

        df.groupby("Transition_ID")

        ["Moisture"]

        .shift(-forecast_horizon)

    )

    df["Future_Ash"] = (

        df.groupby("Transition_ID")

        ["Ash"]

        .shift(-forecast_horizon)

    )

    df["Future_OffSpec"] = (

        (
            abs(
                df["Future_Basis_Weight"]
                -
                df["Target_BW"]
            )

            >

            0.025 * df["Target_BW"]

        )

    ).astype(float)

    df["Future_OffSpec"] = (

        df["Future_OffSpec"]

        .fillna(0)

        .astype(int)

    )

    # ==========================================================
    # CLEAN END OF EACH TRANSITION
    # ==========================================================

    df = df.bfill()

    df = df.ffill()

    # ==========================================================
    # SAVE DATASET
    # ==========================================================

    df.to_csv(

        OUTPUT_FILE,

        index=False

    )

    # ==========================================================
    # SUMMARY
    # ==========================================================

    print("=" * 70)

    print("Honeywell Synthetic Dataset Generated")

    print("=" * 70)

    print("Rows :", len(df))

    print("Columns :", len(df.columns))

    print("Transitions :", df["Transition_ID"].nunique())

    print("Saved To")

    print(OUTPUT_FILE)

    print("=" * 70)

    print(df.head())

    print("=" * 70)

    # ==========================================================
    # MAIN
    # ==========================================================

    if __name__ == "__main__":

        print("Generating Synthetic Dataset...")

        print()

        print("Recipes :", len(RECIPES))

        print("Transitions :", NUM_TRANSITIONS)

        print("Timesteps :", TIMESTEPS)

        print()

        print("Please wait...")

        print()

        # Dataset generation already executed above

        print()

        print("Completed Successfully!")
if __name__ == "__main__":
    generate_dataset()        