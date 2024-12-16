import pandas as pd
from rs3.seq import predict_seq

guide_metadata = pd.read_excel("41588_2021_778_MOESM4_ESM.xlsx", sheet_name=1).set_index("gRNA name")
context_seqs = [
    "AAAC" + g.strip() + "CGGTGT" if g != "Non-Targeting" else "A" * 30 for g in guide_metadata["gRNA sequence"]
]
guide_metadata["rs3_score"] = predict_seq(context_seqs, sequence_tracr="Chen2013")

guide_metadata.to_csv("papalexi_guide_metadata.csv")
