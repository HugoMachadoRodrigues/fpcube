"""Field delineation from v1.1: one connected part per field, and ids kept stable against v1.0."""
import numpy as np
import rasterio
from rasterio.transform import from_origin

from fpcube.fields import connected_within_parcel, stable_ids


def test_parts_split_where_only_a_neighbouring_parcel_connects_them():
    # one pasture patch crossing two parcels; inside parcel 1 its two pieces touch only through parcel 2
    cand = np.array([[1, 1, 1, 1, 1],
                     [0, 0, 1, 0, 0],
                     [1, 1, 1, 1, 1]], bool)
    pidx = np.array([[1, 1, 2, 1, 1],
                     [0, 0, 2, 0, 0],
                     [1, 1, 2, 1, 1]])
    lab = connected_within_parcel(cand, pidx)
    groups = {tuple(sorted(np.unique(pidx[lab == k]))) for k in np.unique(lab[lab > 0])}
    assert groups == {(1,), (2,)}
    assert len(np.unique(lab[lab > 0])) == 5          # four corner pieces of parcel 1 and the column of parcel 2


def test_stable_ids_keep_unchanged_and_largest_part(tmp_path):
    old = np.array([[1, 1, 0, 2, 2, 2],
                    [1, 1, 0, 0, 0, 2],
                    [0, 0, 0, 3, 0, 0]], np.int32)
    p = tmp_path / "old.tif"
    with rasterio.open(p, "w", driver="GTiff", height=3, width=6, count=1, dtype="int32",
                       transform=from_origin(0, 3, 1, 1), crs="EPSG:5070") as d:
        d.write(old, 1)
    new = np.array([[5, 5, 0, 6, 6, 0],              # field 1 unchanged; field 2 split into 6 (2 px) and 7 (1 px)
                    [5, 5, 0, 0, 0, 7],              # field 3 gone
                    [0, 0, 0, 0, 0, 0]], np.int64)
    lab, lin = stable_ids(new, p)
    st = lin.set_index("fid").status.to_dict()
    assert st[1] == "unchanged" and st[2] == "reshaped" and st[3] == "removed"
    assert (lab[0, :2] == 1).all() and (lab[0, 3:5] == 2).all()
    assert lab[1, 5] == 4 and st[4] == "new"                     # next id above the v1.0 maximum
    assert lin.set_index("fid").parent_fid[4] == 2
