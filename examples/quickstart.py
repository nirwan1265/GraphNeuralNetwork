"""Minimal end-to-end example."""
from multigcn import MultiGCN, Config

# 1. Pick algorithms (or all_methods=True for all 16)
model = MultiGCN(
    methods=["GENIE3_ET", "GENIE3_RF", "GRNBOOST2", "CLR", "ARACNE", "MRNET"],
    top_x=0.10,
    epochs=300,
    output_dir="dream4_net1_out",
)

# Check what can actually run on this machine before committing to a long job
print(model.plan())

# 2. Run inference (writes GRN_<ALGO>.csv into <output_dir>/lists/)
model.infer("data/dream4_100_1_expression.csv", skip_failures=True)

# 3. Optionally stack a curated network alongside the inferred ones.
#    Keep this source separate from your evaluation ground truth.
# model.add_prior("REGULONDB", "data/regulondb_edges.csv")

# 4. Train + fuse
scores = model.fit_predict()
model.save()

# 5. Evaluate on the continuous score, and separately on the extracted network
print(model.evaluate("data/dream4_100_1_goldstandard.tsv", k=1.5))
