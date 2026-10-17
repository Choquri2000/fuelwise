# FuelWise - go-to-market

**Live app:** https://fuelwise-car-costs.streamlit.app  |  **Code:** https://github.com/Choquri2000/fuelwise

## Landing copy
**What does your car really cost? Find out in 30 seconds.**
Drop in your fuel log from Fuelly, Drivvo, Fuelio or a spreadsheet. FuelWise works out your true consumption (handling partial fills properly), your cost per km or mile, per month and per year, and tells you if your car suddenly starts drinking more than usual. Your data never leaves your session: nothing is stored, no sign-up.

## Who it helps
- Drivers who log fuel in an app but never look at the analysis.
- People comparing cars, or deciding whether an old car is getting expensive.
- Small-business drivers and gig workers tracking vehicle running costs.
- Anyone who wants a nudge that something (tyre pressure, service) may be worth checking.

## Honest validation plan
1. Test with 5-10 real exports (Fuelly, Drivvo, Fuelio, Google Sheets) from friends/forum volunteers; fix column detection failures first.
2. Post (by the owner, manually) in car-owner communities as a free tool and ask for feedback, watching for "it misread my file" reports.
3. Track simple signals: visits, uploads vs demo use, repeat visits. If few people go past the demo, the file-format friction is the problem.
4. Compare results with each tester's own app numbers to confirm the full-tank logic agrees.
No guarantee of traffic or revenue.

## Free / low-cost monetisation ideas
- Tip jar / "buy me a coffee" link.
- Affiliate links (e.g. tyre pressure gauges, OBD readers) clearly labelled, only where relevant.
- Later: optional paid export (PDF report for tax or resale) if people ask for it.

## Limits to be upfront about
Estimates from user-entered data; anomaly notes are statistical hints, not diagnoses; not tested against every app's export format.
