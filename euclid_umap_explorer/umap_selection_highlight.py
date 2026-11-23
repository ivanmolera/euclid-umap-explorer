from __future__ import annotations

import json
import math

import streamlit.components.v1 as components


SELECTION_HIGHLIGHT_SCRIPT = """
(() => {
    const options = JSON.parse(document.getElementById('umap-focus-data').textContent);
    const host = window.parent;
    const root = host.document;
    const controllers = host.__escopeUmapFocusControllers ||= new Map();
    controllers.get(options.namespace)?.dispose();
    controllers.delete(options.namespace);
    if (!options.focus) return;

    const halo = root.createElement('div');
    halo.className = 'escope-umap-selection-halo';
    halo.dataset.objectId = options.focus.object_id;
    halo.setAttribute('role', 'img');
    halo.setAttribute('aria-label', `Selected object ${options.focus.object_id}`);
    halo.style.cssText = 'position:absolute;width:26px;height:26px;border-radius:50%;'
        + 'background:rgba(160,174,192,0.42);pointer-events:none;z-index:5;'
        + 'transform:translate(-50%,-50%);display:none;';
    let plot = null, resizeObserver = null, disposed = false;
    const pending = new host.MutationObserver(attach);

    function dispose() {
        if (disposed) return;
        disposed = true;
        pending.disconnect();
        resizeObserver?.disconnect();
        plot?.removeListener?.('plotly_relayout', update);
        plot?.removeListener?.('plotly_afterplot', update);
        halo.remove();
        if (controllers.get(options.namespace)?.dispose === dispose) {
            controllers.delete(options.namespace);
        }
    }

    function update() {
        if (disposed) return;
        if (!plot?.isConnected) { dispose(); return; }
        // Use Plotly's axis transforms for both SVG and WebGL, including zoom/pan.
        const xaxis = plot._fullLayout?.xaxis, yaxis = plot._fullLayout?.yaxis;
        if (!xaxis?.d2p || !yaxis?.d2p) return;
        const x = xaxis.d2p(options.focus.x) + xaxis._offset;
        const y = yaxis.d2p(options.focus.y) + yaxis._offset;
        const visible = Number.isFinite(x) && Number.isFinite(y)
            && x >= xaxis._offset && x <= xaxis._offset + xaxis._length
            && y >= yaxis._offset && y <= yaxis._offset + yaxis._length;
        halo.style.display = visible ? 'block' : 'none';
        halo.style.left = `${x}px`;
        halo.style.top = `${y}px`;
    }

    function attach() {
        if (disposed || plot) return;
        const candidate = root.querySelector(`.st-key-${options.chart_key} .js-plotly-plot`);
        const container = candidate?.querySelector('.svg-container');
        if (!container || typeof candidate.on !== 'function') return;
        plot = candidate;
        pending.disconnect();
        container.appendChild(halo);
        plot.on('plotly_relayout', update);
        plot.on('plotly_afterplot', update);
        resizeObserver = new host.ResizeObserver(update);
        resizeObserver.observe(plot);
        update();
    }

    controllers.set(options.namespace, {dispose});
    pending.observe(root.documentElement, {childList: true, subtree: true});
    attach();
})();
"""


def render_umap_2d_selection_highlight(chart_key: str, row: object, *, namespace: str) -> None:
    focus = None
    if row is not None:
        x, y = float(row["umap_1"]), float(row["umap_2"])
        if math.isfinite(x) and math.isfinite(y):
            focus = {"x": x, "y": y, "object_id": str(row["object_id"])}
    options = json.dumps({
        "chart_key": chart_key, "namespace": namespace, "focus": focus,
    }).replace("<", "\\u003c")
    components.html(
        '<script type="application/json" id="umap-focus-data">' + options + '</script>'
        '<script>' + SELECTION_HIGHLIGHT_SCRIPT + '</script>',
        height=0,
    )
