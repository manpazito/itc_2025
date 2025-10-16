from __future__ import annotations
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from pathlib import Path
from typing import Optional, Iterable, Tuple, Dict, Any



def _resolve_ctx_provider(name_or_obj):
    """Accepts a contextily provider object or dotted string like 'CartoDB.PositronNoLabels'."""
    import contextily as ctx
    if not isinstance(name_or_obj, str):
        return name_or_obj
    obj = ctx.providers
    for part in name_or_obj.split("."):
        obj = getattr(obj, part)
    return obj

# Try to import mapping dictionaries (ok if missing)
try:
    from tims_mappings import COLUMN_MAPS, ORDINAL_INT_COLS
except Exception:
    COLUMN_MAPS = {}
    ORDINAL_INT_COLS = []

def plot_crashes_by_category(
    crashes_df: pd.DataFrame | gpd.GeoDataFrame,
    aoi_gdf: gpd.GeoDataFrame,
    category_col: str,
    radius_miles: float = 1.0,
    *,
    lat_col: str = "LATITUDE",
    lon_col: str = "LONGITUDE",
    metric_crs: str = "EPSG:3857",
    geo_crs: str = "EPSG:4326",
    filter_i80: bool = False,
    street_cols: Iterable[str] = ("PRIMARY_RD", "ON_STREET", "ROUTE", "ROUTE_NAME"),
    i80_pattern: Optional[re.Pattern] = None,
    centerlines_gdf: Optional[gpd.GeoDataFrame] = None,
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[str | Path] = None,
    point_size: float = 50,
    point_alpha: float = 0.95,
    legend_title: Optional[str] = None,
    palette: Optional[Dict[str, Any]] = None,
    use_mapped_labels: bool = True,
    show_counts: bool = True,
    add_basemap: bool = True,
    basemap_source: str = "CartoDB.PositronNoLabels",
    buffer_edgecolor: str = "#f50606",
) -> gpd.GeoDataFrame:
    """
    Plot crashes within <radius_miles> of AOI, colored by <category_col>.
    Returns a GeoDataFrame of the crashes inside the buffer (in metric_crs).
    """

    # --- validate inputs ---
    if category_col not in crashes_df.columns:
        raise KeyError(f"'{category_col}' not in crashes_df columns.")
    if aoi_gdf.crs is None:
        raise ValueError("aoi_gdf has no CRS.")

    # --- AOI & buffer (metric CRS) ---
    aoi_metric = aoi_gdf.to_crs(metric_crs)
    aoi_union = aoi_metric.geometry.unary_union
    radius_m = float(radius_miles) * 1609.344
    buffer_gdf = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries([aoi_union], crs=metric_crs).buffer(radius_m),
        crs=metric_crs
    )

    # --- crashes to GeoDataFrame in metric CRS ---
    if isinstance(crashes_df, gpd.GeoDataFrame) and crashes_df.geometry is not None:
        crashes_geo = crashes_df.copy()
        if crashes_geo.crs is None:
            crashes_geo = crashes_geo.set_crs(geo_crs)
        crashes_geo = crashes_geo.to_crs(metric_crs)
    else:
        if lat_col not in crashes_df.columns or lon_col not in crashes_df.columns:
            raise KeyError(f"Need '{lat_col}' and '{lon_col}' to build geometry.")
        tmp = crashes_df.dropna(subset=[lat_col, lon_col]).copy()
        crashes_geo = gpd.GeoDataFrame(
            tmp,
            geometry=gpd.points_from_xy(tmp[lon_col], tmp[lat_col]),
            crs=geo_crs
        ).to_crs(metric_crs)

    # --- optional I-80/ramp filter ---
    if filter_i80:
        if i80_pattern is None:
            i80_pattern = re.compile(
                r"(?:\b(?:i[\s\u2010\u2011-]?80|interstate\s*80|hwy\s*80|highway\s*80|us[\s-]?80)\b)"
                r"|(?:\b(?:ramp|on[\s-]?ramp|off[\s-]?ramp)\b)",
                flags=re.IGNORECASE
            )
        mask = pd.Series(False, index=crashes_geo.index)
        for c in street_cols:
            if c in crashes_geo.columns:
                mask |= crashes_geo[c].astype(str).str.contains(i80_pattern, na=False)
        crashes_geo = crashes_geo.loc[~mask].copy()

    # --- spatial filter: within buffer ---
    within_buffer = gpd.sjoin(
        crashes_geo, buffer_gdf, how="inner", predicate="within"
    ).drop(columns=["index_right"])

    # ------------------- palette & labels -------------------
    raw = within_buffer[category_col]
    ordinal_like = (category_col in ORDINAL_INT_COLS) or pd.api.types.is_numeric_dtype(raw)

    mapdict = COLUMN_MAPS.get(category_col) if use_mapped_labels else None

    def _label(val):
        if mapdict is None:
            return "Unknown" if pd.isna(val) else str(val)
        return mapdict.get(val, mapdict.get(str(val), "Unknown" if pd.isna(val) else str(val)))

    labels = raw.map(_label)

    # order
    if ordinal_like:
        codes_numeric = pd.to_numeric(raw, errors="coerce")
        order_idx = np.argsort(codes_numeric.fillna(np.inf).values)
        ordered = (pd.DataFrame({"code": raw, "label": labels})
                   .iloc[order_idx].drop_duplicates(subset=["label"], keep="first"))
        legend_order = ordered["label"].tolist()
    else:
        legend_order = sorted(pd.Index(labels.unique()).tolist())

    # palette
    if palette is None:
        if ordinal_like:
            cmap = cm.get_cmap("Reds", max(3, len(legend_order)))
            palette = {lab: cmap(i / max(1, len(legend_order) - 1)) for i, lab in enumerate(legend_order)}
        else:
            cmap = cm.get_cmap("Paired", max(3, len(legend_order)))
            palette = {lab: cmap(i / max(1, len(legend_order) - 1)) for i, lab in enumerate(legend_order)}

    # ------------------- plotting -------------------
    fig, ax = plt.subplots(figsize=figsize)

    # extent
    xmin, ymin, xmax, ymax = buffer_gdf.total_bounds
    pad = max(xmax - xmin, ymax - ymin) * 0.05
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal")

    # basemap centerlines (optional)
    if centerlines_gdf is not None:
        cl = centerlines_gdf.copy()
        if cl.crs is None:
            cl = cl.set_crs(geo_crs)
        if cl.crs.to_string() != metric_crs:
            cl = cl.to_crs(metric_crs)
        try:
            cl = gpd.clip(cl, buffer_gdf)
        except Exception:
            pass
        cl.plot(ax=ax, color="lightgrey", linewidth=0.8, zorder=1)

    # AOI + buffer
    gpd.GeoSeries([aoi_union], crs=metric_crs).boundary.plot(
        ax=ax, edgecolor="black", linewidth=2, zorder=3, label="AOI"
    )
    buffer_gdf.boundary.plot(
        ax=ax, edgecolor=buffer_edgecolor, linewidth=1.8, linestyle="--",
        zorder=3, label=f"{radius_miles}-mile radius"
    )

    # points
    gb = within_buffer.assign(_label=labels)
    for lab, group in gb.groupby("_label"):
        group.plot(
            ax=ax,
            markersize=point_size,
            color=palette.get(lab, "gray"),
            alpha=point_alpha,
            edgecolor="black",
            linewidth=0.3,
            label=str(lab),
            zorder=4,
        )

    # legends
    handles, _ = ax.get_legend_handles_labels()
    base_handles = handles[:2]                       # AOI, buffer
    cat_handles = [h for h in handles[2:] if h.get_label() in legend_order]

    legend1 = ax.legend(
        base_handles, ["AOI", f"{radius_miles}-mile radius"],
        loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=8
    )
    ax.add_artist(legend1)

    if show_counts:
        counts = gb["_label"].value_counts()
        cat_labels = [f"{lab}  ({counts.get(lab, 0)})" for lab in legend_order]
    else:
        cat_labels = legend_order

    ax.legend(
        cat_handles, cat_labels,
        loc="upper left", bbox_to_anchor=(1.02, 0.0),
        frameon=True, title=legend_title or category_col, fontsize=8
    )

    ax.set_title(f"Crashes within {radius_miles} miles of AOI — colored by '{category_col}'", pad=10)
    ax.set_axis_off()

    # optional basemap tiles
    if add_basemap:
        try:
            import contextily as ctx
            provider = _resolve_ctx_provider(basemap_source or "CartoDB.PositronNoLabels")
            ctx.add_basemap(
                ax,
                source=provider,
                crs=metric_crs,          # now native
                attribution_size=6,
                alpha=1.0
            )
        except Exception as e:
            print(f"Basemap failed to load: {e}")


    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

    return within_buffer


# small helpers you used elsewhere
def ll_to_xy(t):
    lat, lon = t
    return (lon, lat)

def radius_label(r: float) -> str:
    return f"{r:.2f}".rstrip("0").rstrip(".").replace(".", "_")
