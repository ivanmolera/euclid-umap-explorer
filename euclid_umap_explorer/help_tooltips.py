HELP_TOOLTIP_SCRIPT = r"""
<script>
(() => {
    const host = window.parent;
    const doc = host.document;
    if (host.__escopeHelpTooltips) {
        host.__escopeHelpTooltips.close();
        return;
    }
    let active = null;
    let hideTimer = null;
    let frame = null;
    let nextId = 0;
    const margin = 12;
    const gap = 10;
    const clamp = (value, low, high) => Math.max(low, Math.min(value, high));

    function close() {
        host.clearTimeout(hideTimer);
        if (!active) return;
        const {anchor, popup, portalParent} = active;
        active = null;
        if (typeof popup.hidePopover === 'function' && popup.matches(':popover-open')) popup.hidePopover();
        popup.removeAttribute('data-escope-open');
        anchor.removeAttribute('aria-describedby');
        if (portalParent) {
            if (portalParent.isConnected) portalParent.appendChild(popup);
            else popup.remove();
        }
    }

    function position() {
        frame = null;
        if (!active) return;
        const {anchor, popup} = active;
        if (!anchor.isConnected || !popup.isConnected) return close();
        const rect = anchor.getBoundingClientRect();
        const width = doc.documentElement.clientWidth;
        const height = host.innerHeight;
        if (!rect.width || !rect.height || rect.bottom < 0 || rect.top > height || rect.right <= 0 || rect.left > width) {
            return close();
        }
        popup.style.width = `${Math.min(360, width - 2 * margin)}px`;
        popup.style.maxHeight = `${Math.max(80, height - 2 * margin)}px`;
        const box = popup.getBoundingClientRect();
        const sidebar = anchor.closest('[data-testid="stSidebar"]');
        const sidebarRight = sidebar ? sidebar.getBoundingClientRect().right : rect.right;
        let x;
        let y;
        if (sidebar && width - sidebarRight >= box.width + gap + margin) {
            x = sidebarRight + gap;
            y = rect.top + (rect.height - box.height) / 2;
        } else {
            x = rect.left;
            if (rect.top >= box.height + gap + margin) y = rect.top - box.height - gap;
            else if (height - rect.bottom >= box.height + gap + margin) y = rect.bottom + gap;
            else y = rect.top + (rect.height - box.height) / 2;
        }
        popup.style.left = `${clamp(x, margin, width - box.width - margin)}px`;
        popup.style.top = `${clamp(y, margin, height - box.height - margin)}px`;
    }

    function refresh() {
        if (active && frame === null) frame = host.requestAnimationFrame(position);
    }

    function open(anchor) {
        host.clearTimeout(hideTimer);
        if (active && active.anchor === anchor) return refresh();
        close();
        const popup = anchor.querySelector('.concept-popover');
        if (!popup) return;
        active = {anchor, popup, portalParent: null};
        if (!popup.id) popup.id = `escope-help-${++nextId}`;
        popup.setAttribute('role', 'tooltip');
        popup.setAttribute('tabindex', '0');
        popup.setAttribute('data-escope-open', '');
        anchor.setAttribute('aria-describedby', popup.id);
        popup.style.left = `${margin}px`;
        popup.style.top = `${margin}px`;
        // The top layer escapes Streamlit's clipping containers without moving its DOM.
        if (typeof popup.showPopover === 'function') {
            popup.setAttribute('popover', 'manual');
            popup.showPopover();
        } else {
            active.portalParent = popup.parentElement;
            doc.body.appendChild(popup);
        }
        popup.scrollTop = 0;
        position();
    }

    function anchorFor(target) {
        return target instanceof host.Element ? target.closest('.concept-help') : null;
    }

    doc.addEventListener('pointerover', event => {
        if (active && active.popup.contains(event.target)) {
            host.clearTimeout(hideTimer);
            return;
        }
        const anchor = anchorFor(event.target);
        if (anchor) open(anchor);
    });
    doc.addEventListener('pointerout', event => {
        if (!active || event.pointerType === 'touch') return;
        const {anchor, popup} = active;
        if (!anchor.contains(event.target) && !popup.contains(event.target)) return;
        if (event.relatedTarget && (anchor.contains(event.relatedTarget) || popup.contains(event.relatedTarget))) return;
        host.clearTimeout(hideTimer);
        hideTimer = host.setTimeout(close, 300);
    });
    doc.addEventListener('focusin', event => {
        if (active && active.popup.contains(event.target)) return;
        const anchor = anchorFor(event.target);
        if (anchor) open(anchor);
    });
    doc.addEventListener('focusout', event => {
        if (active && event.relatedTarget &&
            !active.anchor.contains(event.relatedTarget) && !active.popup.contains(event.relatedTarget)) close();
    });
    doc.addEventListener('pointerdown', event => {
        if (active && !active.anchor.contains(event.target) && !active.popup.contains(event.target)) close();
    }, true);
    doc.addEventListener('keydown', event => {
        if (event.key === 'Escape' && active) {
            const anchor = active.anchor;
            close();
            anchor.blur();
        }
    });
    doc.addEventListener('scroll', event => {
        if (active && event.target !== active.popup) refresh();
    }, true);
    doc.addEventListener('load', refresh, true);
    doc.addEventListener('transitionend', refresh, true);
    host.addEventListener('resize', refresh);
    new host.MutationObserver(() => {
        if (active && (!active.anchor.isConnected || !active.popup.isConnected)) close();
    }).observe(doc.body, {childList: true, subtree: true});
    doc.documentElement.classList.add('escope-tooltips-ready');
    host.__escopeHelpTooltips = {close, refresh};
})();
</script>
"""
