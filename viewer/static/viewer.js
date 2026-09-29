// Navigation remains functional without JavaScript; this only improves image viewing.
const dialog = document.querySelector('#image-dialog');
const preview = document.querySelector('#preview-image');
document.querySelectorAll('.image-link').forEach(link => {
  link.addEventListener('click', event => {
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    preview.src = link.href;
    preview.alt = link.querySelector('img').alt;
    dialog.showModal();
  });
});
document.querySelector('.close-preview').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => {
  if (event.target === dialog) dialog.close();
});
dialog.addEventListener('close', () => preview.removeAttribute('src'));
document.querySelectorAll('img[loading="lazy"]').forEach(img => {
  img.addEventListener('error', () => {
    img.closest('figure').classList.add('image-missing');
    img.alt = 'Image unavailable — check the local path or allowed image directories';
  });
});
// A new benchmark may have a different set of model names.
document.querySelector('[data-reset-model]')?.addEventListener('change', event => {
  const model = document.querySelector('#model-select');
  if (model) model.disabled = true;
  event.target.form.querySelector('[name="id"]').value = '';
  event.target.form.requestSubmit();
});
