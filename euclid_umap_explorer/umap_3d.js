// Bundle with Three.js 0.180.0 and esbuild 0.25.11; see frontend/umap3d/package.json.
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const cameraStates = new Map();

export default function({parentElement, data, setStateValue}) {
    let root = parentElement.querySelector('.escope-umap3d');
    if (!root) {
        root = document.createElement('div');
        root.className = 'escope-umap3d';
        parentElement.appendChild(root);
    }
    const saved = cameraStates.get(data.viewKey);
    root.replaceChildren();
    root.style.cssText = `position:relative;width:100%;height:${data.height}px;min-width:0;overflow:hidden;`;
    const style = getComputedStyle(root);
    const foreground = style.getPropertyValue('--st-text-color').trim() || style.color || '#222222';
    const background = style.getPropertyValue('--st-background-color').trim() || '#ffffff';
    const legend = document.createElement('div');
    legend.style.cssText = 'display:flex;gap:12px;flex-wrap:wrap;align-items:center;font-size:12px;min-height:28px;padding:4px 2px;';
    for (const item of data.legend) {
        const label = document.createElement('span');
        label.style.cssText = 'display:inline-flex;gap:5px;align-items:center;';
        const swatch = document.createElement('span');
        swatch.style.cssText = `width:9px;height:9px;background:${item.color};border:1px solid ${foreground};`;
        label.append(swatch, document.createTextNode(item.label));
        legend.append(label);
    }
    root.append(legend);
    const stage = document.createElement('div');
    stage.style.cssText = 'position:relative;width:100%;min-height:240px;touch-action:none;';
    root.append(stage);
    let renderer;
    try {
        renderer = new THREE.WebGLRenderer({antialias: true, preserveDrawingBuffer: true});
    } catch (_) {
        stage.textContent = '3D rendering is unavailable in this browser.';
        return;
    }
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    renderer.setClearColor(background);
    renderer.domElement.setAttribute('aria-label', 'Three-dimensional UMAP');
    renderer.domElement.setAttribute('role', 'img');
    renderer.domElement.style.cssText = 'display:block;width:100%;height:100%;cursor:grab;';
    stage.append(renderer.domElement);
    const tooltip = document.createElement('div');
    tooltip.style.cssText = `display:none;position:absolute;pointer-events:none;z-index:4;padding:8px;max-width:240px;box-sizing:border-box;overflow-wrap:anywhere;white-space:pre-line;background:${background};color:${foreground};border:1px solid ${foreground};border-radius:4px;font-size:12px;`;
    stage.append(tooltip);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, .1, 5000);
    camera.position.set(130, 100, 150);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.minDistance = 12;
    controls.maxDistance = 2500;
    if (saved?.signature === data.signature) {
        camera.position.fromArray(saved.position);
        controls.target.fromArray(saved.target);
    }
    controls.update();
    const saveCamera = () => {
        cameraStates.set(data.viewKey, {signature: data.signature, position: camera.position.toArray(),
            target: controls.target.toArray(), aspect: camera.aspect});
        if (cameraStates.size > 32) cameraStates.delete(cameraStates.keys().next().value);
    };
    const bounds = [0, 1, 2].map(axis => {
        const values = data.points.map(point => point.xyz[axis]);
        let low = Infinity, high = -Infinity;
        for (const value of values) { low = Math.min(low, value); high = Math.max(high, value); }
        return values.length ? [low, high] : [0, 1];
    });
    const center = bounds.map(([low, high]) => (low + high) / 2);
    const scale = 90 / Math.max(...bounds.map(([low, high]) => high - low), 1e-8);
    const position = xyz => xyz.map((value, axis) => (value - center[axis]) * scale);
    const pointPositions = data.points.map(point => position(point.xyz));
    const resources = [];
    function markerTexture(group) {
        const canvas = document.createElement('canvas');
        canvas.width = canvas.height = 64;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = group.color;
        ctx.strokeStyle = group.role === 'Anomaly' ? '#ffcc00' : '#ffffff';
        ctx.lineWidth = 5;
        ctx.beginPath();
        if (group.shape === 'diamond') {
            ctx.moveTo(32, 6); ctx.lineTo(58, 32); ctx.lineTo(32, 58); ctx.lineTo(6, 32); ctx.closePath();
        } else if (group.shape === 'square') {
            ctx.rect(8, 8, 48, 48);
        } else if (group.shape === 'cross' || group.shape === 'x') {
            ctx.save();
            if (group.shape === 'x') { ctx.translate(32, 32); ctx.rotate(Math.PI / 4); ctx.translate(-32, -32); }
            ctx.rect(24, 7, 16, 50); ctx.rect(7, 24, 50, 16);
            ctx.fill();
            if (group.role === 'Anomaly') ctx.stroke();
            ctx.restore();
        } else ctx.arc(32, 32, 25, 0, 2 * Math.PI);
        if (!['cross', 'x'].includes(group.shape)) {
            ctx.fill();
            if (['Canonical', 'Anomaly'].includes(group.role)) ctx.stroke();
        }
        const texture = new THREE.CanvasTexture(canvas);
        resources.push(texture);
        return texture;
    }
    const clouds = [];
    data.groups.forEach((group, groupIndex) => {
        const indices = [];
        data.points.forEach((point, index) => { if (point.group === groupIndex) indices.push(index); });
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(indices.flatMap(i => pointPositions[i]), 3));
        const material = new THREE.PointsMaterial({size: ['Canonical', 'Anomaly'].includes(group.role) ? 15 : 8,
            map: markerTexture(group), alphaTest: .3, transparent: true, opacity: group.role === 'Unknown' ? .72 : 1,
            sizeAttenuation: false});
        const cloud = new THREE.Points(geometry, material);
        cloud.userData.indices = indices;
        scene.add(cloud);
        clouds.push(cloud);
        resources.push(geometry, material);
    });
    function textSprite(text, xyz, size = 4) {
        const canvas = document.createElement('canvas');
        canvas.width = 256; canvas.height = 64;
        const ctx = canvas.getContext('2d');
        ctx.font = '24px sans-serif'; ctx.fillStyle = foreground; ctx.textAlign = 'center';
        ctx.fillText(text, 128, 40);
        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({map: texture, depthTest: false, sizeAttenuation: false});
        const sprite = new THREE.Sprite(material);
        sprite.position.fromArray(xyz); sprite.scale.set(size * .032, size * .008, 1);
        scene.add(sprite); resources.push(texture, material);
    }
    const origin = bounds.map(([low], axis) => (low - center[axis]) * scale - 8);
    for (let axis = 0; axis < 3; axis++) {
        const end = [...origin]; end[axis] = (bounds[axis][1] - center[axis]) * scale + 8;
        const geometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...origin), new THREE.Vector3(...end)]);
        const material = new THREE.LineBasicMaterial({color: foreground, transparent: true, opacity: .45});
        scene.add(new THREE.Line(geometry, material)); resources.push(geometry, material);
        const label = [...end]; label[axis] += 7; textSprite(`UMAP ${axis + 1}`, label, 5);
        for (let tick = 0; tick < 4; tick++) {
            const value = bounds[axis][0] + tick * (bounds[axis][1] - bounds[axis][0]) / 3;
            const xyz = [...origin]; xyz[axis] = (value - center[axis]) * scale;
            xyz[(axis + 1) % 3] -= 4;
            textSprite(Number(value.toPrecision(3)).toString(), xyz, 3);
        }
    }
    const selected = new Set((data.selection?.points || []).map(point => point.customdata[0]));
    const highlightGeometry = new THREE.BufferGeometry();
    const highlightMaterial = new THREE.PointsMaterial({size: 20, color: foreground, sizeAttenuation: false,
        map: markerTexture({color: foreground, shape: 'circle'}), transparent: true, opacity: .35, depthTest: false});
    const highlight = new THREE.Points(highlightGeometry, highlightMaterial);
    highlight.renderOrder = 10; scene.add(highlight); resources.push(highlightGeometry, highlightMaterial);
    const updateHighlight = () => highlightGeometry.setAttribute('position', new THREE.Float32BufferAttribute(
        data.points.flatMap((point, i) => selected.has(point.index) ? pointPositions[i] : []), 3));
    updateHighlight();
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    function hit(event) {
        const rect = renderer.domElement.getBoundingClientRect();
        pointer.set(2 * (event.clientX - rect.left) / rect.width - 1, 1 - 2 * (event.clientY - rect.top) / rect.height);
        raycaster.params.Points.threshold = camera.position.distanceTo(controls.target) * .008;
        raycaster.setFromCamera(pointer, camera);
        const intersections = raycaster.intersectObjects(clouds);
        const match = intersections[0];
        return match ? data.points[match.object.userData.indices[match.index]] : null;
    }
    let down = null;
    function onDown(event) { down = [event.clientX, event.clientY]; }
    function onUp(event) {
        if (!down || Math.hypot(event.clientX - down[0], event.clientY - down[1]) > 5) return;
        down = null;
        const point = hit(event);
        if (!(event.ctrlKey || event.metaKey || event.shiftKey)) selected.clear();
        if (point) {
            if (selected.has(point.index)) selected.delete(point.index); else selected.add(point.index);
        }
        updateHighlight();
        saveCamera();
        setStateValue('selection', {points: [...selected].map(index => ({customdata: [index]}))});
    }
    function onMove(event) {
        const point = hit(event);
        tooltip.style.display = point ? 'block' : 'none';
        if (!point) return;
        tooltip.textContent = `${point.id}\n${point.label}\n${point.xyz.map((v, i) => `UMAP ${i + 1}: ${v.toFixed(4)}`).join('\n')}`;
        const rect = stage.getBoundingClientRect();
        tooltip.style.maxWidth = `${Math.max(80, Math.min(240, rect.width - 16))}px`;
        tooltip.style.left = `${Math.max(4, Math.min(event.clientX - rect.left + 12, rect.width - tooltip.offsetWidth - 8))}px`;
        tooltip.style.top = `${Math.max(4, Math.min(event.clientY - rect.top + 12, rect.height - tooltip.offsetHeight - 8))}px`;
    }
    const hideTooltip = () => { tooltip.style.display = 'none'; };
    const canvas = renderer.domElement;
    canvas.addEventListener('pointerdown', onDown);
    canvas.addEventListener('pointerup', onUp);
    canvas.addEventListener('pointermove', onMove);
    canvas.addEventListener('pointerleave', hideTooltip);
    let checkPixels = true;
    const requestPixelCheck = () => { checkPixels = true; saveCamera(); };
    controls.addEventListener('end', requestPixelCheck);
    let previousAspect = saved?.signature === data.signature ? saved.aspect : null;
    const fitDistance = aspect => 75 / Math.sin(Math.atan(Math.tan(Math.PI / 8) * Math.min(aspect, 1)));
    const resize = () => {
        const width = Math.max(1, root.clientWidth), height = Math.max(240, data.height - legend.offsetHeight);
        stage.style.height = `${height}px`;
        renderer.setSize(width, height); camera.aspect = width / height; camera.updateProjectionMatrix();
        if (previousAspect === null && saved?.signature !== data.signature) {
            camera.position.sub(controls.target).normalize().multiplyScalar(fitDistance(camera.aspect)).add(controls.target);
        } else if (previousAspect) {
            camera.position.sub(controls.target).multiplyScalar(fitDistance(camera.aspect) / fitDistance(previousAspect)).add(controls.target);
        }
        previousAspect = camera.aspect;
        checkPixels = true;
    };
    const observer = new ResizeObserver(resize); observer.observe(root); resize();
    let frame, disposed = false;
    const animate = () => {
        if (disposed) return;
        controls.update(); renderer.render(scene, camera);
        if (checkPixels) {
            // Small DOM diagnostics let browser tests verify actual GPU output without hidden state.
            const gl = renderer.getContext(), width = canvas.width, height = canvas.height;
            const pixels = new Uint8Array(width * height * 4);
            gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
            let visible = 0, hash = 0;
            const stride = Math.max(1, Math.floor(width * height / 16384)) * 4;
            for (let i = 0; i < pixels.length; i += stride) {
                if (Math.abs(pixels[i] - pixels[0]) + Math.abs(pixels[i+1] - pixels[1]) + Math.abs(pixels[i+2] - pixels[2]) > 30) visible++;
                hash = (Math.imul(hash, 31) + pixels[i] + 3 * pixels[i+1] + 7 * pixels[i+2]) | 0;
            }
            canvas.dataset.renderedPixels = JSON.stringify({width, height, visibleSamples: visible, hash});
            checkPixels = false;
        }
        frame = requestAnimationFrame(animate);
    };
    animate();
    return () => {
        disposed = true; cancelAnimationFrame(frame); observer.disconnect();
        saveCamera();
        canvas.removeEventListener('pointerdown', onDown); canvas.removeEventListener('pointerup', onUp);
        canvas.removeEventListener('pointermove', onMove); canvas.removeEventListener('pointerleave', hideTooltip);
        controls.dispose(); resources.forEach(resource => resource.dispose()); renderer.dispose();
    };
}
