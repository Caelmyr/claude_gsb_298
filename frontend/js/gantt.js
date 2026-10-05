/* SVG Gantt chart renderer (dependency-free).

   Usage: renderGantt(container, {
     problem: {resources, tasks, horizon, time_unit},
     assignments: [{task, start, end, resources}],
     mode: 'task' | 'resource',
     title
   })
*/

const GANTT_ROW_H = 34;
const GANTT_BAR_H = 20;
const GANTT_LEFT = 150;
const GANTT_TOP = 30;
const GANTT_PX_PER_UNIT = 24;

function renderGantt(container, opts) {
  const { problem, assignments, mode = 'resource', title } = opts;
  const horizon = problem.horizon || 0;
  const px = GANTT_PX_PER_UNIT;
  const tasks = Object.fromEntries(problem.tasks.map(t => [t.id, t]));
  const resources = Object.fromEntries(problem.resources.map(r => [r.id, r]));

  // rows: task ids or resource ids
  const rowKeys = mode === 'task'
    ? problem.tasks.map(t => t.id)
    : problem.resources.map(r => r.id);
  const rowLabel = (k) => mode === 'task'
    ? `${k} · ${tasks[k]?.name || ''}`.trim()
    : `${k} · ${resources[k]?.name || ''}`.trim();

  const width = GANTT_LEFT + horizon * px + 40;
  const height = GANTT_TOP + rowKeys.length * GANTT_ROW_H + 40;

  container.innerHTML = '';
  const wrap = document.createElement('div');
  wrap.className = 'gantt-wrap';
  wrap.style.position = 'relative';
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);
  svg.style.fontFamily = 'system-ui, sans-serif';
  wrap.appendChild(svg);
  container.appendChild(wrap);

  // tooltip element
  const tip = document.createElement('div');
  tip.className = 'gantt-tooltip';
  tip.style.display = 'none';
  wrap.appendChild(tip);

  function add(tag, attrs) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    svg.appendChild(el);
    return el;
  }

  // header
  if (title) {
    const t = add('text', { x: GANTT_LEFT, y: 18, 'font-size': 13, 'font-weight': 600, fill: '#1f2430' });
    t.textContent = title;
  }

  // time axis grid + labels every 5 units
  const step = Math.max(1, Math.round(horizon / 20));
  for (let t = 0; t <= horizon; t += step) {
    const x = GANTT_LEFT + t * px;
    add('line', { x1: x, y1: GANTT_TOP - 6, x2: x, y2: GANTT_TOP + rowKeys.length * GANTT_ROW_H, stroke: '#eef0f3', 'stroke-width': 1 });
    const lbl = add('text', { x, y: GANTT_TOP - 10, 'font-size': 10, fill: '#6b7280', 'text-anchor': 'middle' });
    lbl.textContent = t;
  }

  // rows
  const rowY = {};
  rowKeys.forEach((k, i) => {
    const y = GANTT_TOP + i * GANTT_ROW_H;
    rowY[k] = y;
    add('line', { x1: GANTT_LEFT, y1: y + GANTT_ROW_H - 8, x2: GANTT_LEFT + horizon * px, y2: y + GANTT_ROW_H - 8, stroke: '#e5e7eb' });
    const lbl = add('text', { x: GANTT_LEFT - 8, y: y + GANTT_ROW_H / 2 + 4, 'font-size': 12, fill: '#374151', 'text-anchor': 'end' });
    lbl.textContent = rowLabel(k);
  });

  // bars
  for (const a of assignments) {
    const x = GANTT_LEFT + a.start * px;
    const w = Math.max(3, (a.end - a.start) * px);
    const color = colorFor(a.task);
    const ys = [];
    if (mode === 'task') {
      ys.push(rowY[a.task]);
    } else {
      for (const r of a.resources) if (rowY[r] !== undefined) ys.push(rowY[r]);
      if (!ys.length) ys.push(rowY[rowKeys[0]]);
    }
    for (const y of ys) {
      if (y === undefined) continue;
      const g = add('g', { class: 'gantt-bar', 'data-task': a.task });
      const rect = add('rect', {
        x, y: y + (GANTT_ROW_H - GANTT_BAR_H) / 2, width: w, height: GANTT_BAR_H,
        rx: 4, fill: color, opacity: 0.85, stroke: 'rgba(0,0,0,.15)',
      });
      if (w > 34) {
        const label = add('text', { x: x + 6, y: y + GANTT_ROW_H / 2 + 4, 'font-size': 11, fill: '#fff', 'font-weight': 600 });
        label.textContent = a.task;
      }
      // interactions
      const show = (ev) => {
        tip.innerHTML = `<b>${escapeHtml(a.task)}</b> · ${escapeHtml(tasks[a.task]?.name || '')}<br>` +
          `开始 ${a.start} → 结束 ${a.end}（工期 ${a.end - a.start} ${problem.time_unit}）<br>` +
          `资源：${escapeHtml((a.resources || []).join(', ') || '—')}`;
        tip.style.display = 'block';
        const rectBox = wrap.getBoundingClientRect();
        tip.style.left = (ev.clientX - rectBox.left + 12) + 'px';
        tip.style.top = (ev.clientY - rectBox.top + 12) + 'px';
      };
      rect.addEventListener('mousemove', show);
      rect.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
    }
  }

  // makespan marker
  const mk = assignments.length ? Math.max(...assignments.map(a => a.end)) : 0;
  if (mk > 0) {
    const x = GANTT_LEFT + mk * px;
    add('line', { x1: x, y1: GANTT_TOP - 6, x2: x, y2: GANTT_TOP + rowKeys.length * GANTT_ROW_H, stroke: '#dc2626', 'stroke-width': 2, 'stroke-dasharray': '4 3' });
    const mkLbl = add('text', { x, y: GANTT_TOP - 12, 'font-size': 11, fill: '#dc2626', 'font-weight': 700, 'text-anchor': 'middle' });
    mkLbl.textContent = '完工时间 ' + mk;
  }
}
