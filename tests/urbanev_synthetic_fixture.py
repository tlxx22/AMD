"""Full synthetic UrbanEV CSVs with the frozen public schema, never observations.

Only the frozen node header is reproduced. Every value below is generated;
4344 timestamps and 275 nodes satisfy the production parser without overrides.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from utils.dataloader_urbanev import (
    EXPECTED_NODE_ORDER_SHA256, EXPECTED_TIMESTAMP_ORDER_SHA256, sequence_sha256,
)

NODE_IDS = (
    "102", "104", "105", "106", "107", "108", "109", "110", "111", "115", "123", "124",
    "202", "204", "205", "206", "208", "212", "213", "214", "215", "216", "217", "223",
    "224", "225", "226", "227", "235", "307", "309", "310", "311", "316", "320", "321",
    "322", "323", "324", "325", "326", "328", "329", "330", "331", "332", "333", "335",
    "340", "343", "346", "347", "348", "406", "408", "501", "502", "506", "508", "511",
    "512", "513", "516", "519", "520", "522", "525", "526", "527", "528", "529", "552",
    "553", "558", "559", "568", "570", "575", "576", "577", "578", "580", "582", "584",
    "585", "587", "588", "589", "592", "594", "595", "596", "598", "601", "628", "631",
    "632", "633", "637", "638", "643", "644", "647", "651", "655", "656", "659", "681",
    "682", "686", "687", "690", "691", "693", "698", "699", "700", "701", "703", "704",
    "705", "706", "708", "709", "710", "711", "712", "715", "716", "718", "719", "724",
    "728", "729", "731", "732", "733", "737", "741", "744", "745", "746", "751", "771",
    "773", "775", "777", "781", "783", "790", "792", "795", "799", "802", "804", "805",
    "809", "832", "842", "844", "848", "849", "851", "852", "855", "858", "862", "881",
    "882", "883", "887", "888", "890", "891", "893", "895", "897", "900", "901", "902",
    "903", "904", "937", "943", "958", "965", "966", "967", "969", "972", "974", "975",
    "977", "979", "981", "982", "983", "984", "986", "987", "988", "989", "991", "996",
    "998", "1000", "1009", "1011", "1015", "1026", "1029", "1031", "1043", "1049", "1060", "1061",
    "1062", "1066", "1067", "1068", "1071", "1072", "1074", "1075", "1076", "1081", "1082", "1083",
    "1085", "1087", "1088", "1090", "1092", "1094", "1095", "1096", "1098", "1099", "1100", "1102",
    "1104", "1106", "1107", "1109", "1110", "1111", "1112", "1113", "1114", "1115", "1119", "1120",
    "1121", "1124", "1125", "1126", "1130", "1134", "1135", "1137", "1138", "1143", "1144", "1149",
    "1154", "1156", "1159", "1162", "1163", "1164", "1166", "1167", "1168", "1172", "1173",
)
TRAIN_END = 3475
VALIDATION_END = 3909


def write_synthetic_urbanev(root, *, perturb=None):
    """Write a fresh fixture; perturb only the named synthetic split."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    clock = pd.date_range("2022-09-01", periods=4344, freq="h")
    assert sequence_sha256(NODE_IDS) == EXPECTED_NODE_ORDER_SHA256
    assert sequence_sha256(tuple(t.isoformat() for t in clock)) == EXPECTED_TIMESTAMP_ORDER_SHA256
    time = np.arange(len(clock), dtype=np.float64)[:, None]
    node = np.arange(len(NODE_IDS), dtype=np.float64)[None, :]
    volume = 10 + time / 128 + (node % 7) / 4 + (time % 24) * (node % 5) / 256
    e_price = 1 + (time % 24) / 8 + (node % 11) / 16
    s_price = 2 + (time % 7) / 4 + (node % 5) / 16
    e_price[:, 0] = 5
    s_price[:, 2] = 0
    # Constant training prices become out-of-range later: no clipping allowed.
    e_price[TRAIN_END:, 0] = 7
    s_price[TRAIN_END:, 2] = -2
    t = time[:, 0]
    weather = np.stack((20 + (t % 48) / 16, 950 + (t % 17) / 8, 50 + (t % 25) / 4), axis=1)
    arrays = {"volume": volume, "e_price": e_price, "s_price": s_price, "weather_central": weather}
    if perturb is not None:
        if perturb not in ("test", "validation"):
            raise ValueError("synthetic perturbation must name validation or test")
        affected = slice(VALIDATION_END, None) if perturb == "test" else slice(TRAIN_END, VALIDATION_END)
        for name, amount in (("volume", 64), ("e_price", 16), ("s_price", -8), ("weather_central", 128)):
            arrays[name][affected] += amount
    timestamps = clock.strftime("%Y-%m-%d %H:%M:%S")
    for name in ("volume", "e_price", "s_price"):
        frame = pd.DataFrame(arrays[name], columns=NODE_IDS)
        frame.insert(0, "time", timestamps)
        frame.to_csv(root / (name + ".csv"), index=False, float_format="%.8f")
    weather_frame = pd.DataFrame(weather, columns=("T", "P", "U"))
    weather_frame.insert(0, "time", timestamps)
    weather_frame.to_csv(root / "weather_central.csv", index=False, float_format="%.8f")
    for name in ("adj.csv", "distance.csv"):
        # The temporal loader validates these headers; it does not consume graphs.
        (root / name).write_text(",".join(NODE_IDS) + "\n", encoding="utf-8")
    pd.DataFrame({"TAZID": NODE_IDS}).to_csv(root / "inf.csv", index=False)
    return {"timestamps": clock, "node_ids": NODE_IDS, **arrays}
