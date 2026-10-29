import numpy as np
import pandas as pd
import pytest

import fuelwise as fw


def log(rows, full=None):
    df = pd.DataFrame(rows, columns=["Date", "Odometer", "Litres", "Price"])
    if full is not None:
        df["Full"] = full
    return df


def test_detect_columns_fuelly_and_drivvo_style():
    m = fw.detect_columns(["fuelup_date", "odometer", "fuel_qty", "price", "partial_fuelup", "missed_fuelup"])
    assert m["date"] == "fuelup_date" and m["volume"] == "fuel_qty" and m["price"] == "price"
    assert m["partial"] == "partial_fuelup" and m["missed"] == "missed_fuelup"
    m = fw.detect_columns(["Odometer", "Date", "Fuel", "Total Cost", "Price/L", "Full tank"])
    assert m["total"] == "Total Cost" and m["price"] == "Price/L" and m["full"] == "Full tank"


def test_to_num():
    assert fw.to_num("1,234.50") == 1234.5
    assert fw.to_num("45,6") == 45.6
    assert fw.to_num("$ 52.30") == 52.3
    assert np.isnan(fw.to_num("abc"))


def test_full_tank_logic_with_partial_fill():
    # full at 1000 (baseline), partial 20 L at 1300, full 20 L at 1600  -> 40 L over 600 km
    df = log([["2024-01-01", 1000, 40, 2.0], ["2024-01-10", 1300, 20, 2.0], ["2024-01-20", 1600, 20, 2.0]],
             full=[1, 0, 1])
    a = fw.analyse(fw.clean_log(df).data)
    assert len(a.intervals) == 1
    iv = a.intervals.iloc[0]
    assert iv.dist_km == 600 and iv.litres == 40
    assert iv.l_per_100km == pytest.approx(40 / 600 * 100)
    assert iv.cost == pytest.approx(80)
    # naive per-row calc would give 20/300=6.7 for the last row; the correct number is 6.67 too only by luck, so
    # check the partial is not treated as its own interval:
    assert fw.analyse(fw.clean_log(df).data).stats["n_intervals"] == 1


def test_first_full_fill_not_counted_and_missed_breaks_chain():
    df = log([["2024-01-01", 1000, 50, 2], ["2024-01-05", 1500, 35, 2], ["2024-01-09", 2000, 35, 2],
              ["2024-01-13", 3000, 35, 2]], full=[1, 1, 1, 1])
    df["Missed"] = [0, 0, 1, 0]
    a = fw.analyse(fw.clean_log(df).data)
    # interval 1000->1500 ok (35 L/500 km = 7); the 1500->2000 interval is broken by the missed flag at 2000
    assert list(a.intervals.dist_km) == [500, 1000]
    assert a.intervals.l_per_100km.iloc[0] == pytest.approx(7.0)


def test_unit_conversion_miles_and_us_gallons():
    # 300 miles on 10 US gal = 30 mpg(US)
    df = log([["2024-01-01", 1000, 10, 3.0], ["2024-01-10", 1300, 10, 3.0]], full=[1, 1])
    a = fw.analyse(fw.clean_log(df, dist_unit="miles", vol_unit="US gal").data)
    l100 = a.intervals.l_per_100km.iloc[0]
    assert fw.conv_consumption(l100, "mpg (US)") == pytest.approx(30.0, rel=1e-3)
    assert fw.conv_consumption(l100, "mpg (UK)") == pytest.approx(30 * 1.20095, rel=1e-3)
    assert fw.conv_consumption(10.0, "km/L") == pytest.approx(10.0)
    # price per US gal $3 -> per litre
    assert fw.clean_log(df, dist_unit="miles", vol_unit="US gal").data.price_per_l.iloc[0] == pytest.approx(3 / 3.785411784)
    assert fw.price_per_volume(1.0, "US gal") == pytest.approx(3.785411784)


def test_uk_gallon_input():
    df = log([["2024-01-01", 0.5, 5, 6.0], ["2024-01-10", 400, 8, 6.0]], full=[1, 1])
    c = fw.clean_log(df, dist_unit="miles", vol_unit="UK gal").data
    assert c.litres.iloc[0] == pytest.approx(5 * 4.54609)


def test_odometer_error_and_impossible_values_flagged():
    df = log([["2024-01-01", 1000, 40, 2], ["2024-01-10", 1500, 35, 2], ["2024-01-15", 900, 35, 2],
              ["2024-01-20", 2000, 900, 2], ["2024-01-25", 2000, 35, 2]], full=[1, 1, 1, 1, 1])
    c = fw.clean_log(df).data
    assert (c.issue != "").sum() == 2 or (c.issue != "").sum() == 3
    assert "odometer" in " ".join(c.issue)
    a = fw.analyse(c)
    assert any(f["kind"] == "data" for f in a.flags)


def test_demo_planted_anomaly_detected():
    a = fw.analyse(fw.clean_log(fw.make_demo()).data)
    rise = [f for f in a.flags if f["kind"] == "rise"]
    assert rise and "may be worth checking" in rise[0]["text"] and "not a diagnosis" in rise[0]["text"]
    iv = a.intervals
    assert iv.l_per_100km.iloc[:15].mean() == pytest.approx(6.8, abs=0.3)
    assert iv.l_per_100km.iloc[-8:].mean() == pytest.approx(8.4, abs=0.3)
    assert a.stats["per_year"] > 0 and a.stats["cost_per_km"] > 0


def test_no_anomaly_on_flat_log():
    d = fw.make_demo()
    d["Fuel (L)"] = (d["Odometer (km)"].diff().fillna(600) * 0.07).round(2)
    d["Full tank"] = 1
    a = fw.analyse(fw.clean_log(d).data)
    assert not [f for f in a.flags if f["kind"] == "rise"]


def test_fuelly_style_csv_and_monthly():
    csv = pd.DataFrame({"fuelup_date": ["2024-01-01", "2024-01-20", "2024-02-10", "2024-03-02"],
                        "odometer": [1000, 1500, 2010, 2500], "fuel_qty": [40, 35, 36, 34],
                        "price": [1.8, 1.9, 1.9, 2.0], "partial_fuelup": [0, 0, 0, 0], "missed_fuelup": [0, 0, 0, 0]})
    c = fw.clean_log(csv)
    a = fw.analyse(c.data)
    assert a.stats["n_intervals"] == 3
    assert len(fw.monthly_costs(c.data)) == 3


def test_tco():
    t = fw.tco(10000, 1000, 500, 300, 1200)
    assert t["per_year"] == 3000 and t["per_month"] == 250 and t["per_km"] == pytest.approx(0.3)


def test_missing_columns_error():
    with pytest.raises(ValueError):
        fw.clean_log(pd.DataFrame({"a": [1], "b": [2]}))


def test_app_end_to_end_demo():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception
    at.checkbox[0].check().run()
    assert not at.exception
    assert any("averages" in s.value for s in at.success)
    assert any("may be worth checking" in w.value for w in at.warning)
    assert len(at.metric) >= 3


def test_csv_safe_neutralises_formulas_only_in_text():
    df = pd.DataFrame({"note": ["=1+1", "ok", None], "litres": [-1.0, 40.0, 41.0]})
    out = fw.csv_safe(df)
    assert out["note"].tolist()[:2] == ["'=1+1", "ok"] and out["litres"].tolist() == [-1.0, 40.0, 41.0]
