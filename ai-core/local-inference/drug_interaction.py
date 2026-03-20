import json
import logging
import os
import csv
import difflib
import requests
from typing import Optional

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/drug_checker.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RIVA.DrugChecker")

# ================================================================
# BRAND → GENERIC (يتحدث من CSV تلقائياً)
# ================================================================
BRAND_TO_GENERIC: dict[str, str] = {
    "panadol": "paracetamol", "paramol": "paracetamol",
    "adol": "paracetamol",    "cataflam": "diclofenac",
    "voltaren": "diclofenac", "brufen": "ibuprofen",
    "advil": "ibuprofen",     "nurofen": "ibuprofen",
    "aspocid": "aspirin",     "amoxil": "amoxicillin",
    "augmentin": "amoxicillin","ciproxin": "ciprofloxacin",
    "flagyl": "metronidazole", "zithromax": "azithromycin",
    "coumadin": "warfarin",    "marevan": "warfarin",
    "capoten": "captopril",    "norvasc": "amlodipine",
    "zocor": "simvastatin",    "crestor": "rosuvastatin",
    "lanoxin": "digoxin",      "cordarone": "amiodarone",
    "lasix": "furosemide",     "glucophage": "metformin",
    "daonil": "glibenclamide", "amaryl": "glimepiride",
    "nexium": "omeprazole",    "losec": "omeprazole",
    "synthroid": "levothyroxine","eltroxin": "levothyroxine",
    "valium": "diazepam",      "xanax": "alprazolam",
}

def _load_medicines_csv(path: str = "data/raw/drug_bank/medicines_list.csv") -> None:
    if not os.path.exists(path):
        logger.warning(f"medicines CSV not found: {path}")
        return
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            brand   = row['brand_name'].lower().strip()
            generic = row['generic_name'].lower().strip()
            BRAND_TO_GENERIC[brand] = generic
    logger.info(f"medicines loaded: {len(BRAND_TO_GENERIC)} names")

_load_medicines_csv()

ALL_KNOWN_NAMES: list[str] = list(BRAND_TO_GENERIC.keys()) + list(set(BRAND_TO_GENERIC.values()))

# ================================================================
# FUZZY MATCHING
# ================================================================
def fuzzy_match(name: str, cutoff: float = 0.8) -> Optional[str]:
    cleaned = name.lower().strip()
    if cleaned in ALL_KNOWN_NAMES:
        return cleaned
    matches = difflib.get_close_matches(cleaned, ALL_KNOWN_NAMES, n=1, cutoff=cutoff)
    if matches:
        logger.info(f"Fuzzy: '{name}' -> '{matches[0]}'")
        return matches[0]
    logger.warning(f"Unknown: '{name}'")
    return None

def smart_normalize(name: str) -> str:
    cleaned = name.lower().strip()
    if cleaned in BRAND_TO_GENERIC:
        g = BRAND_TO_GENERIC[cleaned]
        logger.info(f"Brand: '{name}' -> '{g}'")
        return g
    fuzzy = fuzzy_match(cleaned)
    if fuzzy:
        g = BRAND_TO_GENERIC.get(fuzzy, fuzzy)
        logger.info(f"Fuzzy+Brand: '{name}' -> '{g}'")
        return g
    return cleaned

# ================================================================
# LOCAL INTERACTIONS (fallback لو مفيش CSV)
# ================================================================
LOCAL_INTERACTIONS: dict[tuple[str,str], dict] = {
    ("warfarin","aspirin"):        {"severity":"high",   "effect_ar":"خطر نزيف داخلي شديد"},
    ("warfarin","ibuprofen"):      {"severity":"high",   "effect_ar":"زيادة خطر النزيف"},
    ("warfarin","paracetamol"):    {"severity":"high",   "effect_ar":"تعزيز مفعول وارفارين"},
    ("metformin","alcohol"):       {"severity":"high",   "effect_ar":"خطر الحماض اللاكتيكي"},
    ("digoxin","amiodarone"):      {"severity":"high",   "effect_ar":"سمية قلبية خطيرة"},
    ("simvastatin","amlodipine"):  {"severity":"high",   "effect_ar":"تلف عضلي"},
    ("aspirin","ibuprofen"):       {"severity":"medium", "effect_ar":"قرحة معدة"},
    ("metformin","glibenclamide"): {"severity":"medium", "effect_ar":"هبوط سكر"},
    ("captopril","potassium"):     {"severity":"medium", "effect_ar":"ارتفاع البوتاسيوم"},
    ("calcium","iron"):            {"severity":"low",    "effect_ar":"تقليل امتصاص الحديد"},
}

SEVERITY_LABELS = {"high":"خطير جداً ⛔","medium":"تحذير ⚠️","low":"ملاحظة ℹ️"}
SEVERITY_ORDER  = {"high":0,"medium":1,"low":2}

# ================================================================
# MAIN CLASS
# ================================================================
class DrugInteractionChecker:
    def __init__(self,
                 csv_path:  str = "data/raw/drug_bank/drug_interactions.csv",
                 data_path: str = "ai-core/data/interactions.json") -> None:
        self.local_db = LOCAL_INTERACTIONS.copy()
        self._load_from_csv(csv_path)
        self._load_extended_db(data_path)
        logger.info(f"DrugInteractionChecker ready — {len(self.local_db)} interactions")

    def _load_from_csv(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning(f"drug_interactions CSV not found: {path}")
            return
        count = 0
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row['drug_a'].lower().strip(),
                       row['drug_b'].lower().strip())
                self.local_db[key] = {
                    "severity":  row['severity'],
                    "effect_ar": row['effect_ar']
                }
                count += 1
        logger.info(f"CSV loaded: {count} interactions")

    def _load_extended_db(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as f:
            extra = json.load(f)
        for k, v in extra.items():
            parts = k.split("_")
            if len(parts) == 2:
                self.local_db[(parts[0], parts[1])] = v

    def check_offline(self, new_drug: str, current_drugs: list[str]) -> list[dict]:
        new_g  = smart_normalize(new_drug)
        alerts: list[dict] = []
        for drug in current_drugs:
            ex_g = smart_normalize(drug)
            interaction = (self.local_db.get((new_g, ex_g)) or
                           self.local_db.get((ex_g, new_g)))
            if interaction:
                alerts.append({
                    "drug_a":        new_drug,
                    "drug_b":        drug,
                    "severity":      SEVERITY_LABELS[interaction["severity"]],
                    "severity_code": interaction["severity"],
                    "effect_ar":     interaction["effect_ar"],
                    "source":        "local",
                    "offline":       True
                })
                logger.warning(f"INTERACTION: {new_drug}+{drug} | {interaction['severity'].upper()}")
        if not alerts:
            logger.info(f"SAFE: {new_drug}")
        return sorted(alerts, key=lambda x: SEVERITY_ORDER.get(x["severity_code"], 2))

    def check_online(self, new_drug: str) -> list[dict]:
        generic = smart_normalize(new_drug)
        try:
            url  = f"https://api.fda.gov/drug/label.json?search=openfda.generic_name:{generic}&limit=1"
            resp = requests.get(url, timeout=3)
            results = resp.json().get("results", [])
            if not results: return []
            text = results[0].get("drug_interactions", [""])[0]
            if not text: return []
            logger.info(f"OpenFDA: found interactions for {new_drug}")
            return [{"drug_a": new_drug, "drug_b": "see_text",
                     "severity": "تحذير ⚠️", "severity_code": "medium",
                     "effect_ar": text[:200]+"...",
                     "source": "fda_online", "offline": False}]
        except requests.exceptions.ConnectionError:
            logger.warning("No internet — offline fallback")
            return []
        except Exception as e:
            logger.error(f"OpenFDA error: {e}")
            return []

    def check(self, new_drug: str, current_drugs: list[str]) -> dict:
        logger.info(f"CHECK: '{new_drug}' with {current_drugs}")
        offline = self.check_offline(new_drug, current_drugs)
        online: list[dict] = []
        is_online = False
        try:
            online    = self.check_online(new_drug)
            is_online = bool(online)
        except Exception:
            pass
        all_alerts = offline + online
        result = {
            "new_drug":        new_drug,
            "checked_against": current_drugs,
            "alerts":          all_alerts,
            "is_safe":         not any(a["severity_code"]=="high" for a in all_alerts),
            "source_mode":     "hybrid" if is_online else "offline",
            "alert_count":     len(all_alerts),
        }
        logger.info(f"RESULT: {'SAFE' if result['is_safe'] else 'DANGER'} | {len(all_alerts)} alerts")
        return result

    def is_safe(self, new_drug: str, current_drugs: list[str]) -> bool:
        return self.check(new_drug, current_drugs)["is_safe"]


if __name__ == "__main__":
    checker = DrugInteractionChecker()
    print(f"Total interactions: {len(checker.local_db)}")

    tests = [
        ("panadol",  ["coumadin", "glucophage"]),
        ("warfrine", ["asprin"]),
        ("brufen",   ["aspocid"]),
    ]

    for drug, current in tests:
        result = checker.check(drug, current)
        print(f"\n{drug} + {current}")
        print(f"Safe: {result['is_safe']} | Alerts: {result['alert_count']}")
        for a in result["alerts"]:
            print(f"  {a['severity']}: {a['effect_ar']}")
