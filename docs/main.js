(() => {
  // Resource destinations are intentionally empty until their release.
  document.querySelectorAll('.resource, .footer-resources a').forEach(link => {
    const keepPlaceholder = event => {
      if (!link.getAttribute('href')) event.preventDefault();
    };
    link.addEventListener('click', keepPlaceholder);
    link.addEventListener('auxclick', keepPlaceholder);
  });

  const wall = document.querySelector('#sample-wall');
  if (!wall) return;

  // Stable randomness keeps the sixty samples varied without rearranging them
  // when a viewport changes size. Animation itself stays entirely in CSS.
  let seed = 20261024;
  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };

  const samples = ["38339", "74287", "40510", "15850", "70900", "69292", "1441", "61780", "4912", "86308", "15949", "4512", "55226", "40844", "32696", "57821", "96530", "7063", "49448", "28569", "29520", "55805", "98895", "24439", "13590", "98739", "60279", "6596", "42404", "1913", "14944", "18867", "92580", "37424", "1199", "59478", "69747", "32103", "36052", "89485", "6176", "22481", "59325", "41771", "22558", "3544", "5965", "47900", "47181", "58020", "71902", "59375", "20402", "3012", "90436", "5649", "9507", "32665", "80586", "76932"].map(id => ({id, image: `assets/samples/${id}.webp`, width: 420, height: 594}));
      const rows = [samples.slice(0, 20), samples.slice(20, 40), samples.slice(40, 60)];
      rows.forEach((row, index) => {
        const lane = document.createElement('div');
        lane.className = 'sample-lane';
        lane.style.setProperty('--top', `${-9 + index * 36}vh`);
        lane.style.setProperty('--duration', `${150 + index * 23}s`);
        lane.style.setProperty('--delay', `${-28 - index * 43}s`);
        const group = document.createElement('div');
        group.className = 'sample-group';
        row.forEach(sample => {
          const frame = document.createElement('div');
          frame.className = 'sample';
          frame.dataset.sample = sample.id;
          frame.style.setProperty('--scale', (0.68 + random() * .47).toFixed(3));
          frame.style.setProperty('--offset', `${Math.round((random() - .5) * 38)}px`);
          const image = document.createElement('img');
          image.src = `./${sample.image}`;
          image.alt = '';
          image.width = sample.width;
          image.height = sample.height;
          image.decoding = 'async';
          image.draggable = false;
          frame.append(image);
          group.append(frame);
        });
        lane.append(group, group.cloneNode(true));
        wall.append(lane);
      });

  let heroVisible = true;
  const updateWallPlayback = () => {
    wall.querySelectorAll('.sample-lane').forEach(lane => {
      lane.style.animationPlayState = document.hidden || !heroVisible ? 'paused' : '';
    });
  };
  document.addEventListener('visibilitychange', updateWallPlayback);
  const hero = document.querySelector('.hero');
  if (hero) new IntersectionObserver(entries => {
    heroVisible = entries[0].isIntersecting;
    updateWallPlayback();
  }).observe(hero);
})();

(() => {
  const section = document.querySelector('#evaluation-suite');
  if (!section) return;
  const observer = new IntersectionObserver(entries => {
    if (entries.some(entry => entry.isIntersecting)) {
      section.classList.add('is-visible');
      observer.disconnect();
    }
  }, {threshold: .15});
  observer.observe(section);
})();
