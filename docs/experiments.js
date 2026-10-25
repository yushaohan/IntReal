(() => {
  const explorer = document.querySelector('#experiments-explorer');
  if (!explorer) return;
  const svg = document.querySelector('#results-radar');
  const legend = document.querySelector('#radar-legend');
  const cards = [...document.querySelectorAll('.finding-card')];
  const envelopeKey = document.querySelector('#radar-envelope-key');
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const ns = 'http://www.w3.org/2000/svg';
  const order = ['emu35', 'u1-a3b', 'bagel-sft', 'bagel-sft-rl', 'u1-8b-base', 'u1-8b-sft', 'u1-8b-sft-rl', 'nanobanana2'];
  const strengths = {
    emu35: {key: 'TL', label: 'Best OSS Text Logicality'},
    'u1-a3b': {key: 'CF', label: 'Strong Context Fidelity'},
    'bagel-sft': {key: 'ID', label: 'Strong Image Diversity'},
    'bagel-sft-rl': {key: 'IA', label: 'Best OSS Image Aesthetic'},
    'u1-8b-base': {key: 'CF', label: 'Best Context Fidelity'},
    'u1-8b-sft': {key: 'TR', label: 'Best OSS Text Richness'},
    'u1-8b-sft-rl': {key: 'ID', label: 'Best Image Diversity'},
    nanobanana2: {key: 'TR', label: 'Best Text Richness'}
  };
  let data;
  let models;
  let mode = null;
  let elapsed = 0;
  let lastTime = 0;
  let inView = false;
  let lastFrameKey = '';
  let envelopeGroup;
  let revealPath;
  const profiles = new Map();
  const legendItems = new Map();
  const labels = new Map();

  const element = (tag, attributes = {}, text) => {
    const node = document.createElementNS(ns, tag);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const xy = (index, value) => {
    const angle = index * Math.PI * 2 / 7;
    return [360 + 174 * value / 100 * Math.cos(angle), 265 - 174 * value / 100 * Math.sin(angle)];
  };
  const pathFor = scores => scores.map((value, index) => `${index ? 'L' : 'M'}${xy(index, value).join(',')}`).join(' ') + 'Z';
  const dashFor = model => model.dash.map(value => value * 2.8).join(' ');

  function buildRadar() {
    svg.append(element('circle', {cx: 360, cy: 265, r: 174, class: 'radar-boundary'}));
    data.metrics.forEach((metric, index) => {
      const [x, y] = xy(index, 100);
      const [tx, ty] = xy(index, 135);
      const anchor = [0, 1, 6].includes(index) ? 'start' : [3, 4].includes(index) ? 'end' : 'middle';
      svg.append(element('line', {x1: 360, y1: 265, x2: x, y2: y, class: 'radar-axis'}));
      const group = element('g', {'data-metric': metric.key});
      const full = element('text', {x: tx, y: ty - 6, 'text-anchor': anchor, class: 'radar-axis-label radar-axis-full'});
      const words = metric.label.split(' ');
      full.append(element('tspan', {x: tx}, words[0]), element('tspan', {x: tx, dy: 25}, words.slice(1).join(' ')));
      const short = element('text', {x: tx, y: ty + 8, 'text-anchor': anchor, class: 'radar-axis-label radar-axis-short'}, metric.key);
      group.append(full, short);
      svg.append(group);
      labels.set(metric.key, [full, short]);
    });
    const baseLayer = element('g', {class: 'radar-base-layer'});
    const focusLayer = element('g', {class: 'radar-focus-layer'});
    data.models.forEach(model => {
      const path = pathFor(model.normalizedScores);
      const dash = dashFor(model);
      const base = element('path', {d: path, stroke: model.color, 'stroke-dasharray': dash, class: 'radar-model-base', 'data-profile-id': model.id});
      const title = element('title', {}, `${model.label}: ${data.metrics.map((metric, index) => `${metric.label} ${model.rawScores[index].toFixed(2)}`).join(', ')}`);
      base.append(title);
      baseLayer.append(base);
      const focus = element('g', {class: 'radar-model-focus', 'data-highlight-id': model.id});
      focus.append(
        element('path', {d: path, fill: model.color, class: 'radar-model-fill'}),
        element('path', {d: path, 'stroke-dasharray': dash, class: 'radar-model-halo'}),
        element('path', {d: path, stroke: model.color, 'stroke-dasharray': dash, class: 'radar-model-line'})
      );
      focusLayer.append(focus);
      profiles.set(model.id, focus);
    });
    svg.append(baseLayer, focusLayer);

    const defs = element('defs');
    const mask = element('mask', {id: 'radar-oss-reveal', maskUnits: 'userSpaceOnUse', x: 0, y: 0, width: 760, height: 540});
    mask.append(element('rect', {x: 0, y: 0, width: 760, height: 540, fill: 'black'}));
    revealPath = element('path', {d: pathFor(data.bestOSS.normalizedScores), fill: 'none', stroke: 'white', 'stroke-width': 16, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', pathLength: 1, 'stroke-dasharray': 1, 'stroke-dashoffset': 1});
    mask.append(revealPath);
    defs.append(mask);
    svg.append(defs);
    envelopeGroup = element('g', {class: 'radar-oss-group', mask: 'url(#radar-oss-reveal)'});
    envelopeGroup.append(
      element('path', {d: pathFor(data.bestOSS.normalizedScores), class: 'radar-oss-halo'}),
      element('path', {d: pathFor(data.bestOSS.normalizedScores), class: 'radar-oss-line', 'data-synthetic-profile': 'best-oss-per-metric'})
    );
    svg.append(envelopeGroup);

    order.forEach(id => {
      const model = models.get(id);
      const item = document.createElement('div');
      item.className = 'radar-legend-item';
      item.dataset.model = id;
      const swatch = element('svg', {width: 29, height: 12, viewBox: '0 0 29 12', 'aria-hidden': 'true'});
      swatch.append(element('line', {x1: 0, y1: 6, x2: 29, y2: 6, stroke: model.color, 'stroke-width': 2.8, 'stroke-dasharray': dashFor(model), 'stroke-linecap': 'round'}));
      const name = document.createElement('span');
      name.textContent = model.label;
      item.append(swatch, name);
      legend.append(item);
      legendItems.set(id, item);
    });
  }

  function buildTables() {
    const closed = models.get('nanobanana2');
    const gapRows = document.querySelector('#gap-comparison');
    [0, 1].forEach(index => {
      const row = document.createElement('tr');
      const name = document.createElement('th');
      name.scope = 'row'; name.textContent = data.metrics[index].label;
      row.append(name);
      [data.bestOSS.rawScores[index], closed.rawScores[index]].forEach(value => {
        const cell = document.createElement('td'); cell.textContent = value.toFixed(2); row.append(cell);
      });
      gapRows.append(row);
    });
    const trainingRows = document.querySelector('#training-comparison');
    [{label: 'Mean', index: -1}, ...data.metrics.map((metric, index) => ({label: metric.label, index}))].forEach(metric => {
      const row = document.createElement('tr');
      if (metric.index === -1) row.className = 'mean-row';
      const name = document.createElement('th'); name.scope = 'row'; name.textContent = metric.label; row.append(name);
      data.stageIds.forEach((id, index) => {
        const model = models.get(id);
        const cell = document.createElement('td');
        cell.dataset.stage = index;
        cell.textContent = (metric.index === -1 ? model.mean : model.rawScores[metric.index]).toFixed(2);
        row.append(cell);
      });
      trainingRows.append(row);
    });
  }

  function selectFinding(next) {
    mode = next;
    elapsed = 0;
    lastFrameKey = '';
    explorer.dataset.finding = mode || 'overview';
    document.querySelectorAll('.training-table .is-stage-active').forEach(cell => cell.classList.remove('is-stage-active'));
    cards.forEach(card => {
      const active = card.dataset.finding === mode;
      card.classList.toggle('is-active', active);
      card.querySelector('.finding-trigger').setAttribute('aria-expanded', String(active));
      const panel = card.querySelector('.finding-panel');
      panel.setAttribute('aria-hidden', String(!active));
      panel.inert = !active;
    });
    if (data) render();
  }
  cards.forEach(card => {
    const next = card.dataset.finding;
    const trigger = card.querySelector('.finding-trigger');
    trigger.addEventListener('click', () => selectFinding(mode === next ? null : next));
  });

  function highlight(ids, metricKeys) {
    profiles.forEach((node, id) => node.classList.toggle('is-active', ids.includes(id)));
    legendItems.forEach((node, id) => node.classList.toggle('is-active', ids.includes(id)));
    labels.forEach((nodes, key) => nodes.forEach(node => node.classList.toggle('is-highlighted', metricKeys.includes(key))));
    explorer.dataset.activeModels = ids.join(',');
  }

  function render() {
    const quiet = reducedMotion.matches;
    if (!mode) {
      highlight([], []);
      envelopeGroup.style.opacity = '0';
      revealPath.setAttribute('stroke-dashoffset', '1');
      envelopeKey.classList.remove('is-visible');
      envelopeKey.setAttribute('aria-hidden', 'true');
      explorer.dataset.envelopeProgress = '0';
      explorer.dataset.phase = 'overview';
      lastFrameKey = 'overview';
    } else if (mode === 'strengths') {
      const phase = quiet ? 0 : Math.floor(elapsed / 2300) % order.length;
      const id = order[phase];
      if (lastFrameKey !== `strengths-${id}`) {
        const model = models.get(id), detail = strengths[id];
        highlight([id], [detail.key]);
        document.querySelector('#model-strength').style.setProperty('--model-color', model.color);
        document.querySelector('#strength-model').textContent = model.label;
        document.querySelector('#strength-metric').textContent = detail.label;
        document.querySelector('#strength-score').textContent = model.rawScores[data.metrics.findIndex(item => item.key === detail.key)].toFixed(2);
        explorer.dataset.phase = phase;
        lastFrameKey = `strengths-${id}`;
      }
      envelopeGroup.style.opacity = '0';
      revealPath.setAttribute('stroke-dashoffset', '1');
      envelopeKey.classList.remove('is-visible');
      envelopeKey.setAttribute('aria-hidden', 'true');
      explorer.dataset.envelopeProgress = '0';
    } else if (mode === 'gap') {
      const progress = quiet ? 1 : Math.max(0, Math.min(1, (elapsed - 1150) / 1200));
      const eased = 1 - Math.pow(1 - progress, 3);
      const phase = progress === 0 ? 'closed' : progress < 1 ? 'drawing' : 'both';
      if (lastFrameKey !== `gap-${phase}`) {
        highlight(['nanobanana2'], ['TR', 'TL']);
        explorer.dataset.phase = phase;
        lastFrameKey = `gap-${phase}`;
      }
      envelopeGroup.style.opacity = progress > 0 ? '1' : '0';
      revealPath.setAttribute('stroke-dashoffset', String(1 - eased));
      envelopeKey.classList.toggle('is-visible', progress > 0);
      envelopeKey.setAttribute('aria-hidden', String(progress === 0));
      explorer.dataset.envelopeProgress = eased.toFixed(3);
    } else {
      const phase = quiet ? 2 : Math.floor(elapsed / 2600) % data.stageIds.length;
      const id = data.stageIds[phase];
      if (lastFrameKey !== `training-${phase}`) {
        highlight(quiet ? data.stageIds : [id], phase === 0 ? ['QC', 'CF'] : phase === 1 ? ['TR', 'TL', 'IA', 'CF'] : ['IA', 'ID', 'CF']);
        document.querySelectorAll('.training-table [data-stage]').forEach(cell => cell.classList.toggle('is-stage-active', quiet || Number(cell.dataset.stage) === phase));
        explorer.dataset.phase = phase;
        lastFrameKey = `training-${phase}`;
      }
      envelopeGroup.style.opacity = '0';
      revealPath.setAttribute('stroke-dashoffset', '1');
      envelopeKey.classList.remove('is-visible');
      envelopeKey.setAttribute('aria-hidden', 'true');
      explorer.dataset.envelopeProgress = '0';
    }
  }

  reducedMotion.addEventListener('change', () => { lastFrameKey = ''; if (data) render(); });
  new IntersectionObserver(entries => { inView = entries[0].isIntersecting; }, {threshold: .1}).observe(explorer);
  function tick(time) {
    const delta = Math.min(time - lastTime, 100);
    lastTime = time;
    if (mode && inView && !document.hidden) {
      elapsed += delta;
      render();
    }
    requestAnimationFrame(tick);
  }

  fetch('./assets/experiments/data.json')
    .then(response => { if (!response.ok) throw new Error('Results unavailable'); return response.json(); })
    .then(value => {
      data = value;
      models = new Map(data.models.map(model => [model.id, model]));
      buildRadar();
      buildTables();
      render();
      explorer.setAttribute('aria-busy', 'false');
      requestAnimationFrame(tick);
    })
    .catch(() => {
      explorer.setAttribute('aria-busy', 'false');
      document.querySelector('#experiments-error').hidden = false;
    });
})();
