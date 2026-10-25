(() => {
  const explorer = document.querySelector('#dataset-explorer');
  if (!explorer) return;

  const chart = document.querySelector('#dataset-chart');
  const choices = document.querySelector('#category-choices');
  const viewport = document.querySelector('#cover-viewport');
  const track = document.querySelector('#cover-track');
  const dialog = document.querySelector('#sample-dialog');
  const fullImage = document.querySelector('#sample-page-full');
  const previewImage = document.querySelector('#sample-page-preview');
  const page = document.querySelector('#sample-page');
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const svgNS = 'http://www.w3.org/2000/svg';
  let data;
  let category;
  let currentSamples = [];
  let currentIndex = 0;
  let returnFocus;
  let manualPauseUntil = 0;
  let dragging = false;
  let hovered = false;
  let focused = false;
  let inView = false;
  let cycleWidth = 0;
  let previousTime = 0;

  const point = (radius, angle) => {
    const radians = angle * Math.PI / 180;
    return [180 + radius * Math.cos(radians), 180 + radius * Math.sin(radians)];
  };
  const ring = (start, end) => {
    const [ax, ay] = point(153, start), [bx, by] = point(153, end);
    const [cx, cy] = point(91, end), [dx, dy] = point(91, start);
    const large = end - start > 180 ? 1 : 0;
    return `M${ax},${ay}A153,153 0 ${large} 1 ${bx},${by}L${cx},${cy}A91,91 0 ${large} 0 ${dx},${dy}Z`;
  };

  function createChart() {
    let start = -90;
    // Keep the category order and hues from the original distribution figure.
    ['knowledge', 'story', 'tutorial', 'planning'].forEach(key => {
      const item = data.categories.find(value => value.key === key);
      const end = start + item.percent * 3.6;
      const middle = (start + end) / 2;
      const slice = document.createElementNS(svgNS, 'g');
      slice.classList.add('chart-slice');
      slice.dataset.category = item.key;
      slice.dataset.dx = (Math.cos(middle * Math.PI / 180) * 7).toFixed(2);
      slice.dataset.dy = (Math.sin(middle * Math.PI / 180) * 7).toFixed(2);
      slice.setAttribute('role', 'button');
      slice.setAttribute('tabindex', '0');
      slice.setAttribute('aria-label', `${item.label}, ${item.percent.toFixed(2)} percent`);
      slice.setAttribute('aria-controls', 'cover-viewport');
      const path = document.createElementNS(svgNS, 'path');
      path.setAttribute('d', ring(start, end));
      path.setAttribute('fill', item.color);
      const title = document.createElementNS(svgNS, 'title');
      title.textContent = `${item.label} · ${item.percent.toFixed(2)}%`;
      slice.append(title, path);
      slice.addEventListener('click', () => selectCategory(item.key));
      slice.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectCategory(item.key); }
      });
      chart.append(slice);
      start = end;
    });
    data.categories.forEach(item => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'category-choice';
      button.dataset.category = item.key;
      button.style.setProperty('--category-color', item.color);
      button.textContent = item.label;
      button.setAttribute('aria-controls', 'cover-viewport');
      button.addEventListener('click', () => selectCategory(item.key));
      choices.append(button);
    });
  }

  function createCover(sample, index, duplicate) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'cover-card';
    button.dataset.id = sample.id;
    button.setAttribute('aria-label', `Open ${sample.title}, ${sample.source}`);
    if (duplicate) button.tabIndex = -1;
    const imageWrap = document.createElement('span');
    imageWrap.className = 'cover-image-wrap';
    const image = document.createElement('img');
    image.src = `./${sample.cover}`;
    image.alt = '';
    image.width = 480;
    image.height = 679;
    image.loading = 'lazy';
    image.decoding = 'async';
    image.draggable = false;
    const zoom = document.createElement('span');
    zoom.className = 'cover-open-indicator';
    zoom.textContent = '↗';
    zoom.setAttribute('aria-hidden', 'true');
    imageWrap.append(image, zoom);
    const title = document.createElement('span');
    title.className = 'cover-title';
    title.textContent = sample.title;
    const source = document.createElement('span');
    source.className = 'cover-source';
    source.textContent = sample.source;
    button.append(imageWrap, title, source);
    button.addEventListener('click', () => openSample(index, button));
    return button;
  }

  function measureCycle() {
    cycleWidth = track.querySelector('.cover-group')?.getBoundingClientRect().width || 0;
  }

  function selectCategory(key) {
    if (category?.key === key) return;
    category = data.categories.find(item => item.key === key);
    const sourceQueues = category.sources.map(source => data.samples.filter(sample => sample.category === key && sample.source === source));
    currentSamples = [];
    while (sourceQueues.some(queue => queue.length)) {
      sourceQueues.forEach(queue => { if (queue.length) currentSamples.push(queue.shift()); });
    }
    explorer.style.setProperty('--selected-color', category.color);
    document.querySelector('#active-category').textContent = category.label;
    document.querySelector('#active-sources').textContent = category.sources.join(' · ');
    document.querySelector('#active-share').replaceChildren(document.createTextNode(category.percent.toFixed(2)), Object.assign(document.createElement('span'), {textContent: '%'}));
    viewport.setAttribute('aria-label', `${category.label} sample covers`);
    [...chart.children, ...choices.children].forEach(button => {
      const active = button.dataset.category === key;
      button.setAttribute('aria-pressed', String(active));
      const path = button.querySelector('path');
      if (path) path.style.transform = active ? `translate(${button.dataset.dx}px, ${button.dataset.dy}px)` : '';
    });
    const groups = [false, true].map(duplicate => {
      const group = document.createElement('div');
      group.className = 'cover-group';
      if (duplicate) group.setAttribute('aria-hidden', 'true');
      currentSamples.forEach((sample, index) => group.append(createCover(sample, index, duplicate)));
      return group;
    });
    track.replaceChildren(...groups);
    viewport.scrollLeft = 0;
    focused = false;
    measureCycle();
    document.querySelector('#category-announcement').textContent = `${category.label}, ${category.percent.toFixed(2)} percent. Showing sample covers from ${category.sources.join(', ')}.`;
  }

  viewport.addEventListener('pointerenter', event => { if (event.pointerType !== 'touch') hovered = true; });
  viewport.addEventListener('pointerleave', () => { hovered = false; });
  viewport.addEventListener('focusin', () => { focused = true; });
  viewport.addEventListener('focusout', () => { focused = viewport.contains(document.activeElement); });
  // Let a touch drag or horizontal trackpad gesture settle before resuming.
  viewport.addEventListener('pointerdown', event => {
    if (event.pointerType === 'touch') { dragging = true; manualPauseUntil = performance.now() + 8000; }
  });
  const endDrag = () => {
    if (dragging) { dragging = false; manualPauseUntil = performance.now() + 8000; }
  };
  document.addEventListener('pointerup', endDrag);
  document.addEventListener('pointercancel', endDrag);
  viewport.addEventListener('wheel', event => {
    if (Math.abs(event.deltaX) > Math.abs(event.deltaY)) manualPauseUntil = performance.now() + 8000;
  }, {passive: true});
  new ResizeObserver(measureCycle).observe(viewport);

  function showSample(index) {
    currentIndex = (index + currentSamples.length) % currentSamples.length;
    const sample = currentSamples[currentIndex];
    document.querySelector('#sample-title').textContent = sample.title;
    document.querySelector('#sample-meta').textContent = `${category.label} · ${sample.source}`;
    document.querySelector('#sample-image-error').hidden = true;
    page.classList.remove('is-loaded');
    previewImage.src = `./assets/samples/${sample.id}.webp`;
    fullImage.alt = `${sample.title}: the full interleaved text and image sample.`;
    fullImage.src = `./${sample.body}`;
    document.querySelector('#sample-reader').scrollTop = 0;
  }
  function openSample(index, trigger) {
    returnFocus = trigger;
    showSample(index);
    document.body.classList.add('modal-open');
    dialog.showModal();
    document.querySelector('#sample-close').focus({preventScroll: true});
  }
  fullImage.addEventListener('load', () => page.classList.add('is-loaded'));
  fullImage.addEventListener('error', () => { document.querySelector('#sample-image-error').hidden = false; });
  document.querySelector('#sample-close').addEventListener('click', () => dialog.close());
  document.querySelector('#sample-previous').addEventListener('click', () => showSample(currentIndex - 1));
  document.querySelector('#sample-next').addEventListener('click', () => showSample(currentIndex + 1));
  dialog.addEventListener('keydown', event => {
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault(); showSample(currentIndex + (event.key === 'ArrowLeft' ? -1 : 1));
    }
  });
  dialog.addEventListener('click', event => {
    if (event.target !== dialog) return;
    const bounds = dialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
  });
  dialog.addEventListener('close', () => {
    document.body.classList.remove('modal-open');
    returnFocus?.focus({preventScroll: true});
  });

  new IntersectionObserver(entries => { inView = entries[0].isIntersecting; }, {threshold: .05}).observe(explorer);
  function tick(time) {
    const elapsed = Math.min(time - previousTime, 60);
    previousTime = time;
    if (inView && !reducedMotion.matches && time >= manualPauseUntil && !dragging && !hovered && !focused && !dialog.open && !document.hidden && cycleWidth) {
      viewport.scrollLeft += elapsed * .027;
      if (viewport.scrollLeft >= cycleWidth) viewport.scrollLeft -= cycleWidth;
    }
    requestAnimationFrame(tick);
  }

  fetch('./assets/explore/data.json')
    .then(response => { if (!response.ok) throw new Error('Sample data unavailable'); return response.json(); })
    .then(value => {
      data = value;
      createChart();
      selectCategory('planning');
      explorer.setAttribute('aria-busy', 'false');
      requestAnimationFrame(tick);
    })
    .catch(() => {
      const status = document.querySelector('#gallery-status');
      if (status) status.textContent = 'Samples could not be loaded. Please refresh to try again.';
      explorer.setAttribute('aria-busy', 'false');
    });
})();
