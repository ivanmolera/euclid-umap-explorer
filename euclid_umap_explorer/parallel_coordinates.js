export default function(component) {
    const {parentElement, data, setStateValue} = component;
    let wrapper = parentElement.querySelector('.escope-parallel');
    if (!wrapper) {
        wrapper = document.createElement('div');
        wrapper.className = 'escope-parallel';
        wrapper.style.cssText = 'width:100%;overflow-x:auto;overflow-y:hidden;';
        parentElement.appendChild(wrapper);
    }
    const chart = document.createElement('div');
    chart.className = 'escope-parallel-chart';
    chart.setAttribute('role', 'img');
    chart.setAttribute('aria-label', data.label);
    wrapper.replaceChildren(chart);
    let disposed = false, timer;
    const styles = getComputedStyle(wrapper);
    const foreground = styles.getPropertyValue('--st-text-color').trim() || styles.color;
    const background = styles.getPropertyValue('--st-background-color').trim() || '#ffffff';
    const figure = structuredClone(data.figure);
    figure.layout.font = {color: foreground, family: styles.fontFamily};
    figure.layout.paper_bgcolor = background;
    figure.layout.plot_bgcolor = background;
    figure.data[0].labelfont.color = foreground;
    figure.data[0].tickfont.color = foreground;
    figure.data[0].line.colorbar.tickfont = {color: foreground, size: 11};
    const colourTitle = figure.data[0].line.colorbar.title;
    figure.data[0].line.colorbar.title = {text: colourTitle.text || colourTitle, font: {color: foreground, size: 11}};
    const width = () => Math.max(wrapper.clientWidth, figure.data[0].dimensions.length * 115 + 170);
    figure.layout.width = width();
    let last = JSON.stringify(data.brush);
    Plotly.newPlot(chart, figure.data, figure.layout, {displayModeBar: false, responsive: false}).then(() => {
        if (disposed) return;
        chart.on('plotly_restyle', () => {
            clearTimeout(timer);
            timer = setTimeout(() => {
                if (disposed) return;
                const dimensions = chart.data[0].dimensions;
                const brush = {order: dimensions.map(d => d.name), ranges: {}};
                for (const dimension of dimensions) {
                    if (dimension.constraintrange?.length) brush.ranges[dimension.name] = dimension.constraintrange;
                }
                const encoded = JSON.stringify(brush);
                if (encoded !== last) {
                    last = encoded;
                    setStateValue('brush', brush);
                }
            }, 180);
        });
    }).catch(() => {
        if (!disposed) chart.textContent = 'Could not display the multivariate chart.';
    });
    const observer = new ResizeObserver(() => {
        if (!disposed && chart.layout && Math.abs(chart.layout.width - width()) > 1) Plotly.relayout(chart, {width: width()});
    });
    observer.observe(wrapper);
    return () => {
        disposed = true;
        clearTimeout(timer);
        observer.disconnect();
        Plotly.purge(chart);
    };
}
