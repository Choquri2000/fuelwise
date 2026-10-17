# FuelWise

Privacy-first fuel and running-cost analyser for car owners. Upload a fuel log (Fuelly, Drivvo, Fuelio, any spreadsheet CSV) or type fill-ups in, and get true consumption, cost per km/mile, monthly and yearly cost, price trend, and careful "may be worth checking" notes.

- **Private:** processed in memory only, nothing stored, no AI/API calls.
- **Units:** km / miles; litres / US gal / UK gal; show L/100km, mpg (US), mpg (UK) or km/L; any currency symbol.
- **Demo:** tick *Use demo data* (synthetic log with a planted consumption jump). `demo_fuel_log.csv` is the same data.

## How consumption is calculated
Full-tank-to-full-tank. For each full fill-up, fuel = that fill + any partial fills since the previous full fill-up; distance = odometer difference between the two full fill-ups. The first full fill is only a starting point. A "missed fill-up" flag breaks the chain; implausible intervals (outside 1.5-40 L/100km or far from the car's median) are excluded and reported.

## Anomaly checks (wording is deliberately cautious)
- Persistent step up in consumption (best two-segment fit, at least 15% and statistically clear) -> "may be worth checking" tyre pressure, servicing, driving mix. Not a diagnosis.
- Single spikes vs the car's own rolling baseline.
- Odometer going backwards/duplicated, volume <= 0 or > 250 L, price far from the rest -> row ignored and listed.

## Run
```
pip install -r requirements.txt
streamlit run app.py
pytest
```

## Limits
Only tested on synthetic data and hand-made files in the common formats, not real exports from every app. Column detection is heuristic; check the "How your columns were read" panel. Estimates, not mechanical or financial advice.
