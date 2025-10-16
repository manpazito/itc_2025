from __future__ import annotations
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path
import warnings

from tims_mappings import apply_tims_mappings, COLUMN_MAPS, ORDINAL_INT_COLS

# Silent the warnings, nothing stops me
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*Downcasting object dtype arrays on .fillna.*",
)
warnings.filterwarnings("ignore", category=FutureWarning, module="tims_mappings")


def _resolve_ctx_provider(name_or_obj):
    """Return a contextily provider whether given an object or 'CartoDB.PositronNoLabels' string."""
    import contextily as ctx

    if not isinstance(name_or_obj, str):
        return name_or_obj
    obj = ctx.providers
    for part in name_or_obj.split("."):
        obj = getattr(obj, part)
    return obj

def plot_crashes_by_category(
    crashes_df,
    aoi_gdf,
    category_col,
    radius_miles=1.0,
    *,
    lat_col="LATITUDE",
    lon_col="LONGITUDE",
    metric_crs="EPSG:3857",
    geo_crs="EPSG:4326",
    filter_i80=False,
    street_cols=("PRIMARY_RD", "ON_STREET", "ROUTE", "ROUTE_NAME"),
    i80_pattern=None,
    centerlines_gdf=None,
    figsize=(15, 15),
    save_path=None,
    point_size=50,
    point_alpha=0.95,
    legend_title=None,
    palette=None,
    use_mapped_labels=True,
    show_counts=True,
    add_basemap=True,
    basemap_source="CartoDB.PositronNoLabels",
    buffer_edgecolor="#305CDE",
):
    """
    Plot crashes within a given radius of the AOI, colored by a category column.
    Returns the crashes inside the buffer (in metric_crs).
    """
    # --- basic checks ---
    if category_col not in crashes_df.columns:
        raise KeyError(f"'{category_col}' not in crashes_df columns.")
    if aoi_gdf.crs is None:
        raise ValueError("aoi_gdf has no CRS.")
    radius_miles = round(radius_miles, 4) if str(radius_miles) else radius_miles

    # --- AOI & buffer (metric CRS) ---
    aoi_metric = aoi_gdf.to_crs(metric_crs)
    aoi_union = aoi_metric.geometry.unary_union
    radius_m = float(radius_miles) * 1609.344
    buffer_gdf = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries([aoi_union], crs=metric_crs).buffer(radius_m),
        crs=metric_crs,
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
            tmp, geometry=gpd.points_from_xy(tmp[lon_col], tmp[lat_col]), crs=geo_crs
        ).to_crs(metric_crs)

    # --- optional I-80/ramp filter ---
    if filter_i80:
        if i80_pattern is None:
            i80_pattern = re.compile(
                r"(?:\b(?:i[\s\u2010\u2011-]?80|interstate\s*80|hwy\s*80|highway\s*80|us[\s-]?80)\b)"
                r"|(?:\b(?:ramp|on[\s-]?ramp|off[\s-]?ramp)\b)",
                flags=re.IGNORECASE,
            )
        mask = pd.Series(False, index=crashes_geo.index)
        for c in street_cols:
            if c in crashes_geo.columns:
                mask |= crashes_geo[c].astype(str).str.contains(i80_pattern, na=False)
        crashes_geo = crashes_geo.loc[~mask].copy()

    # --- keep only points within the buffer ---
    within_buffer = gpd.sjoin(
        crashes_geo, buffer_gdf, how="inner", predicate="within"
    ).drop(columns=["index_right"])

    # ------------------- labels & color palette -------------------
    raw = within_buffer[category_col]
    ordinal_like = (category_col in ORDINAL_INT_COLS) or pd.api.types.is_numeric_dtype(
        raw
    )

    mapdict = COLUMN_MAPS.get(category_col) if use_mapped_labels else None

    def _label(val):
        if mapdict is None:
            return "Unknown" if pd.isna(val) else str(val)
        return mapdict.get(
            val, mapdict.get(str(val), "Unknown" if pd.isna(val) else str(val))
        )

    labels = raw.map(_label)

    # Legend order
    if ordinal_like:
        codes_numeric = pd.to_numeric(raw, errors="coerce")
        order_idx = np.argsort(codes_numeric.fillna(np.inf).values)
        ordered = (
            pd.DataFrame({"code": raw, "label": labels})
            .iloc[order_idx]
            .drop_duplicates(subset=["label"], keep="first")
        )
        legend_order = ordered["label"].tolist()
    else:
        legend_order = sorted(pd.Index(labels.unique()).tolist())

    # Palette
    if palette is None:
        if ordinal_like:
            cmap = cm.get_cmap("Reds", max(3, len(legend_order)))
        else:
            cmap = cm.get_cmap("Paired", max(3, len(legend_order)))
        palette = {
            lab: cmap(i / max(1, len(legend_order) - 1))
            for i, lab in enumerate(legend_order)
        }

    # ------------------- plotting -------------------
    fig, ax = plt.subplots(figsize=figsize)

    # Extent with a small pad
    xmin, ymin, xmax, ymax = buffer_gdf.total_bounds
    pad = max(xmax - xmin, ymax - ymin) * 0.05
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal")

    # Optional centerlines layer
    if centerlines_gdf is not None:
        cl = centerlines_gdf.copy()
        if cl.crs is None:
            cl = cl.set_crs(geo_crs)
        if cl.crs is None or cl.crs.to_string() != metric_crs:
            cl = cl.to_crs(metric_crs)
        try:
            cl = gpd.clip(cl, buffer_gdf)
        except Exception:
            pass
        cl.plot(ax=ax, color="#545454", linewidth=2, zorder=1)

    # AOI boundary + buffer ring
    gpd.GeoSeries([aoi_union], crs=metric_crs).boundary.plot(
        ax=ax, edgecolor="#f50606", linewidth=3, zorder=3, label="AOI"
    )
    buffer_gdf.boundary.plot(
        ax=ax,
        edgecolor=buffer_edgecolor,
        linewidth=1.8,
        linestyle="--",
        zorder=3,
        label=f"{radius_miles}-mile radius",
    )

    # Crash points by label
    gb = within_buffer.assign(_label=labels)
    for lab, group in gb.groupby("_label"):
        group.plot(
            ax=ax,
            markersize=point_size,
            color=(palette.get(lab, "gray") if isinstance(palette, dict) else "gray"),
            alpha=point_alpha,
            edgecolor="black",
            linewidth=0.3,
            label=str(lab),
            zorder=4,
        )

    # Legends (split base vs categories)
    handles, _ = ax.get_legend_handles_labels()
    base_handles = handles[:2]  # AOI, buffer
    cat_handles = [h for h in handles[2:] if h.get_label() in legend_order]

    legend1 = ax.legend(
        base_handles,
        ["AOI", f"{radius_miles}-mile radius"],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        fontsize=8,
    )
    ax.add_artist(legend1)

    if show_counts:
        counts = gb["_label"].value_counts()
        cat_labels = [f"{lab}  ({counts.get(lab, 0)})" for lab in legend_order]
    else:
        cat_labels = legend_order

    ax.legend(
        cat_handles,
        cat_labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.0),
        frameon=True,
        title=legend_title or category_col,
        fontsize=8,
    )

    ax.set_title(
        f"Crashes within {radius_miles} miles of AOI — colored by '{category_col}'",
        pad=10, 
        fontsize= 20
    )
    ax.set_axis_off()

    # Optional basemap tiles
    if add_basemap:
        try:
            import contextily as ctx

            provider = _resolve_ctx_provider(
                basemap_source or "CartoDB.PositronNoLabels"
            )
            ctx.add_basemap(
                ax, source=provider, crs=metric_crs, attribution_size=6, alpha=1.0
            )
        except Exception as e:
            print(f"Basemap failed to load: {e}")

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

    return within_buffer


# Small helpers used elsewhere
def ll_to_xy(t):
    """(lat, lon) -> (lon, lat) tuple for GeoPandas builders."""
    lat, lon = t
    return (lon, lat)


def radius_label(r):
    """Format a radius number into a short label (e.g., 1.5 -> '1_5')."""
    return f"{r:.2f}".rstrip("0").rstrip(".").replace(".", "_")


def aoi_dashboard(
    crashes_gdf,
    *,
    title="Crash Characteristics within AOI Buffer",
    top_n_pcfcat=10,
    save_path=None,
    figsize=(20, 15),
):
    """
    Build a 3x3 dashboard of crash summaries for the AOI buffer.

    Row 1: Severity • PCF_VIOL_CATEGORY (top N) • Collision Type
    Row 2: Weather • Lighting • Day of Week
    Row 3: Alcohol Involved • CHP Shift • Road Condition 1

    Works with numeric or string TIMS codes; uses tims_mappings if available.
    """
    needed = [
        "COLLISION_SEVERITY",
        "TYPE_OF_COLLISION",
        "PCF_VIOL_CATEGORY",
        "WEATHER_1",
        "LIGHTING",
        "DAY_OF_WEEK",
        "ALCOHOL_INVOLVED",
        "CHP_SHIFT",
        "ROAD_COND_1",
    ]
    missing = [c for c in needed if c not in crashes_gdf.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # 1) Readable copy (prefer apply_tims_mappings when present)
    if apply_tims_mappings is not None:
        df = apply_tims_mappings(crashes_gdf)[needed].copy()
    else:
        df = crashes_gdf[needed].copy()
        for col in needed:
            if col in COLUMN_MAPS:
                df[col] = df[col].map(COLUMN_MAPS[col]).fillna(df[col])

    # Fill only object columns with a friendly string
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].fillna("Not Stated")

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    axes = axes.flatten()

    # Helper: horizontal bar with counts on bars
    def _barh(ax, counts, title_, palette_name="Set2", wrap=26):
        import textwrap

        labels = [textwrap.fill(str(s), wrap) for s in counts.index]
        colors = sns.color_palette(palette_name, n_colors=len(labels))
        sns.barplot(
            y=labels,
            x=counts.values,
            hue=labels,
            palette=colors,
            dodge=False,
            legend=False,
            ax=ax,
        )
        ax.set_title(title_)
        ax.set_xlabel("Count")
        ax.set_ylabel("")
        for p, v in zip(ax.patches, counts.values):
            ax.text(
                p.get_width() + 0.3,
                p.get_y() + p.get_height() / 2,
                int(v),
                va="center",
                fontsize=9,
            )

    # 1) Severity (numeric-safe)
    sev_code_order = [0, 4, 3, 2, 1]  # PDO, Possible, Minor, Severe, Fatal
    sev_labels_order = ["PDO", "Possible", "Minor", "Severe", "Fatal"]
    sev_counts_num = pd.to_numeric(
        crashes_gdf["COLLISION_SEVERITY"], errors="coerce"
    ).value_counts()
    sev_vals = [int(sev_counts_num.get(code, 0)) for code in sev_code_order]
    sev_colors = sns.color_palette("Reds", n_colors=len(sev_labels_order))
    sns.barplot(
        y=sev_labels_order,
        x=sev_vals,
        hue=sev_labels_order,
        palette=sev_colors,
        dodge=False,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Crash Severity Distribution")
    axes[0].set_xlabel("Count")
    axes[0].set_ylabel("")
    for p, v in zip(axes[0].patches, sev_vals):
        axes[0].text(
            p.get_width() + 0.3,
            p.get_y() + p.get_height() / 2,
            v,
            va="center",
            fontsize=9,
        )

    # 2) PCF_VIOL_CATEGORY (top N)
    pcfcat_counts = df["PCF_VIOL_CATEGORY"].value_counts().head(top_n_pcfcat)
    _barh(axes[1], pcfcat_counts, f"Top {top_n_pcfcat} PCF Violation Categories")

    # 3) Collision Type
    type_counts = df["TYPE_OF_COLLISION"].value_counts()
    _barh(axes[2], type_counts, "Collision Type Distribution")

    # 4) Weather
    weather_counts = df["WEATHER_1"].value_counts()
    _barh(axes[3], weather_counts, "Weather at Time of Collision")

    # 5) Lighting
    light_counts = df["LIGHTING"].value_counts()
    _barh(axes[4], light_counts, "Lighting Conditions")

    # 6) Day of Week (numeric-safe)
    day_code_order = [1, 2, 3, 4, 5, 6, 7]  # Sun..Sat
    day_labels = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ]
    dow_counts_num = pd.to_numeric(
        crashes_gdf["DAY_OF_WEEK"], errors="coerce"
    ).value_counts()
    dow_vals = [int(dow_counts_num.get(code, 0)) for code in day_code_order]
    dow_colors = sns.color_palette("Set2", n_colors=len(day_labels))
    sns.barplot(
        x=day_labels,
        y=dow_vals,
        hue=day_labels,
        palette=dow_colors,
        dodge=False,
        legend=False,
        ax=axes[5],
    )
    axes[5].set_title("Crashes by Day of Week")
    axes[5].set_xlabel("")
    axes[5].set_ylabel("Count")
    for t in axes[5].get_xticklabels():
        t.set_rotation(30)
    for p, v in zip(axes[5].patches, dow_vals):
        axes[5].text(
            p.get_x() + p.get_width() / 2,
            p.get_height(),
            v,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 7) Alcohol Involved (Y/blank)
    alcohol_raw = crashes_gdf["ALCOHOL_INVOLVED"].astype(str).str.upper()
    alcohol_counts = pd.Series(
        {"Yes": int((alcohol_raw == "Y").sum()), "No": int((alcohol_raw != "Y").sum())}
    )
    colors = sns.color_palette("Set2", n_colors=2)
    sns.barplot(
        y=alcohol_counts.index,
        x=alcohol_counts.values,
        hue=alcohol_counts.index,
        palette=colors,
        dodge=False,
        legend=False,
        ax=axes[6],
    )
    axes[6].set_title("Alcohol Involved")
    axes[6].set_xlabel("Count")
    axes[6].set_ylabel("")
    for p, v in zip(axes[6].patches, alcohol_counts.values):
        axes[6].text(
            p.get_width() + 0.3,
            p.get_y() + p.get_height() / 2,
            int(v),
            va="center",
            fontsize=9,
        )

    # ---------- 8) MVIW (Motor Vehicle Involved With) ----------
    mviw_map = COLUMN_MAPS["MVIW"]

    # Build labels using mapping (robust to codes or already-labeled strings)
    mviw_raw = crashes_gdf["MVIW"].astype(str).str.strip()
    mviw_labels = mviw_raw.map(mviw_map).fillna(mviw_raw)

    # Count & (optionally) limit to top N
    mviw_counts = mviw_labels.value_counts().head(10)  # remove .head(10) to show all
    _labels = mviw_counts.index.astype(str)

    # Colors & plot (use hue to avoid seaborn palette warnings)
    colors = sns.color_palette("Set2", n_colors=len(_labels))
    sns.barplot(
        y=_labels,
        x=mviw_counts.values,
        hue=_labels,
        palette=colors,
        dodge=False,
        legend=False,
        ax=axes[7],
    )
    axes[7].set_title("Motor Vehicle Involved With (MVIW)")
    axes[7].set_xlabel("Count")
    axes[7].set_ylabel("")
    for p, v in zip(axes[7].patches, mviw_counts.values):
        axes[7].text(
            p.get_width() + 0.3,
            p.get_y() + p.get_height() / 2,
            int(v),
            va="center",
            fontsize=9,
        )

    # 9) Road Condition 1
    rc_counts = df["ROAD_COND_1"].value_counts()
    _barh(axes[8], rc_counts, "Road Condition (Primary)")

    # Layout / save
    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
    return fig, axes
