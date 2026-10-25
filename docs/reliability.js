(() => {
  const section = document.querySelector('#metric-reliability');
  const dialog = document.querySelector('#reliability-dialog');
  if (!section || !dialog) return;
  const title = document.querySelector('#reliability-dialog-title');
  const image = document.querySelector('#reliability-dialog-image');
  let opener;
  let previousOverflow = '';

  section.querySelectorAll('.reliability-figure').forEach(button => {
    button.addEventListener('click', () => {
      const source = button.querySelector('img');
      opener = button;
      title.textContent = button.closest('.reliability-card').querySelector('h3').textContent;
      image.src = source.src;
      image.alt = source.alt;
      previousOverflow = document.body.style.overflow;
      document.body.style.overflow = 'hidden';
      dialog.showModal();
    });
  });
  dialog.querySelector('.reliability-dialog-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    if (event.target !== dialog) return;
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  });
  dialog.addEventListener('close', () => {
    document.body.style.overflow = previousOverflow;
    opener?.focus({preventScroll: true});
  });
})();
