# Short, plot-friendly label dictionaries + an apply() helper for TIMS crash data.
from __future__ import annotations
import pandas as pd

# ---------- Small helpers ----------
NOT_STATED = "Not Stated"

def _yes_blank_map(x):
    # Columns defined as "Y or blank"
    return { "Y": "Yes", "": "No", None: "No" }.get(x, "Yes" if str(x).upper()=="Y" else "No")

def _dash_or_blank(x):
    return True if (x is None or str(x).strip() in {"", "-", "--"}) else False

# ---------- Crash-level mappings ----------
DAY_OF_WEEK = {
     1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"
}
CHP_SHIFT = {
     1: "06:00–13:59", 2: "14:00–21:59", 3: "22:00–05:59", 4: "CHP NS", 5: "Not CHP"
}
SPECIAL_COND = {
    "1": "Schoolbus on Rdwy", "2": "State Univ / SFIA", "3": "Schoolbus off Rdwy",
    "4": "Offroad (Unimpr.)", "5": "Vista/Rest/Scale/Inspect", "6": "Other Public Access",
    "0": "None", "-": NOT_STATED
}
BEAT_TYPE = {
    1: "CHP State Hwy", 2: "CHP Cnty Line", 3: "CHP Cnty Area",
    4: "Schoolbus City Rdwy", 5: "Schoolbus not Rdwy",
    6: "Offroad (Unimpr.)", 7: "Vista/Rest/Scale", 8: "Other Public Access",
    0: "Not CHP"
}
CHP_BEAT_TYPE = {
    "1": "Interstate", "2": "US Hwy", "3": "State Route", "4": "Cnty Line",
    "5": "Cnty Area", "A": "Safety Services", "S": "Admin (900s)", "0": "Not CHP",
    # Contract city (string codes)
    "6": "US Hwy (City)", "7": "State Rt (City)", "8": "Cnty Line (City)", "9": "Cnty Area (City)"
}
CHP_BEAT_CLASS = { 1: "CHP Primary", 2: "CHP Other", 0: "Not CHP" }

DIRECTION = { "N": "North", "E": "East", "S": "South", "W": "West" }
INTERSECTION = { "Y": "Intersection", "N": "Not Intersection" }

WEATHER = {
    "A": "Clear", "B": "Cloudy", "C": "Raining", "D": "Snowing", "E": "Fog",
    "F": "Other", "G": "Wind", "-": NOT_STATED
}

STATE_HWY_IND = { "Y": "State Hwy", "N": "Not State Hwy" }

LOCATION_TYPE = { "H": "Highway", "I": "Intersection", "R": "Ramp/Collector", "-": "Not State Hwy" }
RAMP_INTERSECTION = {
    "1": "Ramp Exit (last 50')", "2": "Mid Ramp", "3": "Ramp Entry (first 50')",
    "4": "Ramp-related (≤100')", "5": "Intersection", "6": "Intxn-related (≤250')",
    "7": "Highway", "8": "Not State Hwy", "-": NOT_STATED
}
SIDE_OF_HWY = { "N": "NB", "S": "SB", "E": "EB", "W": "WB" }

# Ordinal (cast to int)
COLLISION_SEVERITY = {
    1: "Fatal", 2: "Severe", 3: "Minor", 4: "Possible", 0: "PDO"
}

PRIMARY_COLL_FACTOR = {
    "A": "Code Violation", "B": "Other Improper Driving", "C": "Other Than Driver",
    "D": "Unknown", "E": "Fell Asleep", "-": NOT_STATED
}
PCF_CODE_OF_VIOL = {
    "B": "Business & Professions", "C": "Vehicle", "H": "City H&S", "I": "City Ord",
    "O": "County Ord", "P": "Penal", "S": "Streets & Highways", "W": "Welfare & Inst",
    "-": NOT_STATED
}
# NOTE: category keys contain leading zeros -> keep as strings
PCF_VIOL_CATEGORY = {
    "01": "DUI (Driver/Bicyclist)", "02": "Impeding Traffic", "03": "Unsafe Speed",
    "04": "Following Too Closely", "05": "Wrong Side", "06": "Improper Passing",
    "07": "Unsafe Lane Change", "08": "Improper Turning", "09": "Auto ROW",
    "10": "Ped ROW", "11": "Ped Violation", "12": "Signals & Signs",
    "13": "Hazardous Parking", "14": "Lights", "15": "Brakes", "16": "Other Equip",
    "17": "Other Hazardous Viol", "18": "Other Than Driver/Ped",
    "19": "—", "20": "—", "21": "Unsafe Start/Backing",
    "22": "Other Improper Driving", "23": "Ped/Other DUI",
    "24": "Fell Asleep",
    "00": "Unknown", "-": NOT_STATED
}

HIT_AND_RUN = { "F": "Felony", "M": "Misdemeanor", "N": "No H&R" }

TYPE_OF_COLLISION = {
    "A": "Head-On", "B": "Sideswipe", "C": "Rear End", "D": "Broadside",
    "E": "Hit Object", "F": "Overturned", "G": "Veh/Ped", "H": "Other", "-": NOT_STATED
}

MVIW = {
    "A": "Non-Collision", "B": "Pedestrian", "C": "Other MV", "D": "MV Other Rdwy",
    "E": "Parked MV", "F": "Train", "G": "Bicycle", "H": "Animal", "I": "Fixed Object",
    "J": "Other Object", "0": "Non-Collision + Add’l Obj", "1": "Ped + Add’l Obj",
    "2": "Other MV + Add’l Obj", "3": "MV Other Rdwy + Add’l Obj",
    "4": "Parked MV + Add’l Obj", "5": "Train + Add’l Obj", "6": "Bicycle + Add’l Obj",
    "7": "Animal + Add’l Obj", "8": "Fixed Obj + Add’l Obj", "9": "Other Obj + Add’l Obj",
    "-": NOT_STATED
}

PED_ACTION = {
    "A": "No Ped", "B": "Xwalk @Intxn", "C": "Xwalk Not @Intxn", "D": "Crossing Not in Xwalk",
    "E": "In Road/Shoulder", "F": "Not in Road", "G": "Near School Bus", "-": NOT_STATED
}

ROAD_SURFACE = { "A": "Dry", "B": "Wet", "C": "Snow/Ice", "D": "Slippery", "-": NOT_STATED }
ROAD_COND = {
    "A": "Holes/Ruts", "B": "Loose Material", "C": "Obstruction",
    "D": "Const/Repair Zone", "E": "Reduced Width", "F": "Flooded",
    "G": "Other", "H": "No Unusual", "-": NOT_STATED
}
LIGHTING = {
    "A": "Daylight", "B": "Dusk/Dawn", "C": "Dark w/ Lights",
    "D": "Dark, No Lights", "E": "Dark, Lights Out", "-": NOT_STATED
}
CONTROL_DEVICE = {
    "A": "Functioning", "B": "Not Functioning", "C": "Obscured", "D": "None", "-": NOT_STATED
}

# “Y or blank” flags (map to Yes/No)
Y_BLANK_COLS = [
    "PEDESTRIAN_ACCIDENT","BICYCLE_ACCIDENT","MOTORCYCLE_ACCIDENT",
    "TRUCK_ACCIDENT","NOT_PRIVATE_PROPERTY","ALCOHOL_INVOLVED"
]

# ---------- Master mapping dict (column -> mapping) ----------
COLUMN_MAPS: dict[str, dict] = {
    "DAY_OF_WEEK": DAY_OF_WEEK,
    "CHP_SHIFT": CHP_SHIFT,
    "SPECIAL_COND": SPECIAL_COND,
    "BEAT_TYPE": BEAT_TYPE,
    "CHP_BEAT_TYPE": CHP_BEAT_TYPE,
    "CHP_BEAT_CLASS": CHP_BEAT_CLASS,
    "DIRECTION": DIRECTION,
    "INTERSECTION": INTERSECTION,
    "WEATHER_1": WEATHER,
    "WEATHER_2": WEATHER,   # some files use WEATHER_2
    "STATE_HWY_IND": STATE_HWY_IND,
    "LOCATION_TYPE": LOCATION_TYPE,
    "RAMP_INTERSECTION": RAMP_INTERSECTION,
    "SIDE_OF_HWY": SIDE_OF_HWY,
    "COLLISION_SEVERITY": COLLISION_SEVERITY,  # ordinal (int)
    "PRIMARY_COLL_FACTOR": PRIMARY_COLL_FACTOR,
    "PCF_CODE_OF_VIOL": PCF_CODE_OF_VIOL,
    "PCF_VIOL_CATEGORY": PCF_VIOL_CATEGORY,    # keep keys as strings (leading zeros)
    "HIT_AND_RUN": HIT_AND_RUN,
    "TYPE_OF_COLLISION": TYPE_OF_COLLISION,
    "MVIW": MVIW,
    "PED_ACTION": PED_ACTION,
    "ROAD_SURFACE": ROAD_SURFACE,
    "ROAD_COND_1": ROAD_COND,
    "ROAD_COND_2": ROAD_COND,
    "LIGHTING": LIGHTING,
    "CONTROL_DEVICE": CONTROL_DEVICE,
    # Y/blank handled separately
}

# ---------- Columns that should be cast to int (ordinal) if present ----------
ORDINAL_INT_COLS = [
    "COLLISION_SEVERITY",
    "DAY_OF_WEEK",
    "CHP_SHIFT",
    # counts are numeric already but we ensure dtype
    "NUMBER_KILLED","NUMBER_INJURED","PARTY_COUNT",
    "COUNT_SEVERE_INJ","COUNT_VISIBLE_INJ","COUNT_COMPLAINT_PAIN",
    "COUNT_PED_KILLED","COUNT_PED_INJURED",
    "COUNT_BICYCLIST_KILLED","COUNT_BICYCLIST_INJURED",
    "COUNT_MC_KILLED","COUNT_MC_INJURED"
]

def apply_tims_mappings(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with code values mapped to short, human-readable labels.
    - Keeps string codes with letters or leading zeros as strings.
    - Casts ordinal columns to int where safe.
    - Maps Y/blank flags to Yes/No."""
    out = df.copy()

    # Normalize dash/blank to explicit code where needed
    for col, mapping in COLUMN_MAPS.items():
        if col in out.columns:
            ser = out[col]
            # preserve raw code for matching (strings vs ints)
            if ser.dtype.kind in "ifc":
                # numeric: map using numeric keys; handle NaNs -> Not Stated
                out[col] = ser.map(mapping).fillna(ser).fillna(NOT_STATED)
            else:
                # strings: treat "-" or blanks as Not Stated
                out[col] = ser.where(~ser.map(_dash_or_blank).fillna(False), "-")
                out[col] = out[col].map(mapping).fillna(out[col])

    # Y/blank columns -> Yes/No
    for col in Y_BLANK_COLS:
        if col in out.columns:
            out[col] = out[col].apply(_yes_blank_map)

    # Ordinal ints
    for col in ORDINAL_INT_COLS:
        if col in out.columns:
            # Convert cleanly; ignore non-numeric with errors='coerce'
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    return out